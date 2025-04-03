import duckdb
import pathlib
import yaml
import argparse
import os
import sys
import time

class Config:
    def __init__(self, c: dict):
        self.data_path = pathlib.Path(c['data_path'])
        self.output_path = pathlib.Path(c['output_path'])
        self.db_path = c.get('db_path')
        self.aggregation_hours = c.get('aggregation_hours', 6)

        # Files (required)
        self.inpatient_file = self.data_path / c['inpatient_file']
        self.labs_file = self.data_path / c['labs_file']
        self.diagnoses_file = self.data_path / c['diagnoses_file']
        self.procedures_file = self.data_path / c['procedures_file']
        self.medications_file = self.data_path / c['medications_file']

        # Limits
        self.diagnoses_limit = c['diagnoses_limit']
        self.procedures_limit = c['procedures_limit']
        self.labs_limit = c['labs_limit']
        self.medications_limit = c['medications_limit']

        # Vitals (optional)
        self.vitals_dbp_file = self._opt_path(c.get('vitals_dbp_file'))
        self.vitals_o2_device_file = self._opt_path(c.get('vitals_o2_device_file'))
        self.o2_device_mapping_file = self._opt_path(c.get('o2_device_mapping_file'))
        self.vitals_pulse_file = self._opt_path(c.get('vitals_pulse_file'))
        self.vitals_resp_rate_file = self._opt_path(c.get('vitals_resp_rate_file'))
        self.vitals_sbp_file = self._opt_path(c.get('vitals_sbp_file'))
        self.vitals_sp_o2_file = self._opt_path(c.get('vitals_sp_o2_file'))
        self.vitals_temp_file = self._opt_path(c.get('vitals_temp_file'))

        # Checking for file existence
        for name, path in vars(self).items():
            if isinstance(path, pathlib.Path) and not path.exists():
                raise FileNotFoundError(f"❌ [Config] File not found: {name} → {path}")

    def _opt_path(self, filename: str | None):
        return self.data_path / filename if filename else None

def load_dataset_config(path: str):
    cc = []

    with open(path, 'r') as f:
        conf = yaml.safe_load(f)

        for c in conf['datasets']:
            cc.append(Config(c))

    return cc

def process_data(conf: Config):
    print("📦 Config parameters:")
    for k, v in vars(conf).items():
        print(f"  • {k}: {v}")

    db_path = conf.db_path if conf.db_path else ':memory:'
    conn = duckdb.connect(db_path)
    create_schema(conn)

    print('🔧 Loading raw inpatient data')
    load_raw_inpatient_data(conn, conf.inpatient_file)

    create_static_data_vocab(conn)
    create_deduplicated_inpatient_table(conn)
    create_static_feature_array(conn)
    create_encounters_period_table(conn)

    print('🔧 Processing labs')
    load_labs(conn, conf.labs_file)
    split_labs_values_into_groups(conn, conf.labs_limit)
    create_labs_values_vocab(conn)
    create_labs_features(conn, conf.aggregation_hours)

    print('🔧 Processing vitals')
    vitals_paths = {
        "vitals_dbp_file": conf.vitals_dbp_file,
        "vitals_o2_device_file": conf.vitals_o2_device_file,
        "o2_device_mapping_file": conf.o2_device_mapping_file,
        "vitals_pulse_file": conf.vitals_pulse_file,
        "vitals_resp_rate_file": conf.vitals_resp_rate_file,
        "vitals_sbp_file": conf.vitals_sbp_file,
        "vitals_sp_o2_file": conf.vitals_sp_o2_file,
        "vitals_temp_file": conf.vitals_temp_file,
    }
    load_vitals(conn, vitals_paths)
    create_vitals_values_vocab(conn)
    create_dynamic_vitals_features(conn, conf.aggregation_hours)

    print('🔧 Processing diagnoses')
    load_diagnoses(conn, conf.diagnoses_file, conf.data_path)
    create_diagnoses_values_vocab(conn, conf.diagnoses_limit)
    create_diagnoses_features(conn, conf.aggregation_hours)

    print('🔧 Processing procedures')
    load_procedures(conn, conf.procedures_file)
    create_procedures_values_vocab(conn, conf.procedures_limit)
    create_procedures_features(conn, conf.aggregation_hours)

    print('🔧 Processing medications')
    load_medications(conn, conf.medications_file)
    create_medications_values_vocab(conn, conf.medications_limit)
    create_medications_features(conn, conf.aggregation_hours)

    print('🔧 Creating dynamic features arrays')
    create_dynamic_features_arrays(conn)

    print('🔧 Save the processed data')
    pathlib.Path(conf.output_path).mkdir(parents=True, exist_ok=True)

    print('🔧 Save vocabs')
    save_static_data_vocab(conn, conf.output_path)
    save_dynamic_data_vocab(conn, conf.output_path)

    print('🔧 Save the processed data to parquet file')
    save_data(conn, conf.aggregation_hours ,conf.output_path)
    save_target(conn, conf.output_path)

    print('🔧 Finishing...')
    if db_path != ':memory:':
        conn.execute("VACUUM")
    conn.close()
    print('🔧 DONE')

def create_schema(conn: duckdb.DuckDBPyConnection):
    # static data vocab
    conn.query(f"DROP SEQUENCE IF EXISTS static_data_vocab_id_seq CASCADE;")
    conn.query(f"CREATE SEQUENCE static_data_vocab_id_seq;")
    conn.query(f"DROP TABLE IF EXISTS static_data_vocab;")
    conn.query(f"""
        CREATE TABLE IF NOT EXISTS static_data_vocab (
            id INTEGER PRIMARY KEY DEFAULT nextval('static_data_vocab_id_seq'),
            type VARCHAR,
            description VARCHAR,
            code VARCHAR,
            lower_value FLOAT DEFAULT 0,
            upper_value FLOAT DEFAULT 0,
            encounters BIGINT
        );
    """)
    # dynamic data vocab
    conn.query(f"DROP SEQUENCE IF EXISTS dynamic_data_vocab_id_seq CASCADE;")
    conn.query(f"CREATE SEQUENCE dynamic_data_vocab_id_seq;")
    conn.query(f"DROP TABLE IF EXISTS dynamic_data_vocab;")
    conn.query(f"""
        CREATE TABLE IF NOT EXISTS dynamic_data_vocab (
            id INT PRIMARY KEY DEFAULT nextval('dynamic_data_vocab_id_seq'),
            type VARCHAR,
            description VARCHAR,
            code VARCHAR,
            value_group INT,
            lower_value FLOAT DEFAULT 0,
            upper_value FLOAT DEFAULT 0,
            encounters BIGINT
        );
    """)

def load_raw_inpatient_data(conn: duckdb.DuckDBPyConnection, file_path: str):
    conn.query(f"DROP TABLE IF EXISTS raw_inpatient;")
    conn.query(f"""
        CREATE TABLE raw_inpatient AS
        WITH data AS (
            SELECT 
                StudyID AS patient_id,
                PatientEncounterID AS encounter_id,
                HospitalAdmitDTS AS admitted,
                HospitalDischargeDTS AS discharged,
                HospitalServiceCD AS service_id,
                HospitalServiceDSC AS service_desc,
                DischargeCategory AS disposition_desc,
                MeansOfArrivalCD AS ma_id,
                MeansOfArrivalDSC AS ma_desc,
                IsFemale AS is_female,
                Age AS age,
                RaceEthnicity AS race,
                IsMarried AS is_married,
                SpeaksEnglish AS is_speak_english,
                HasAtLeastSomeCollege AS is_graduated,
                SDI_score AS sdi,
                HasSmoked AS is_smoked,
                CurrentlyDrinks AS is_drunk,
                DaysFromLastHospitalization AS days_from_last,
                Within30 AS in_30,
                AnyICU AS was_in_icu,
                LastDepartment AS last_department
            FROM read_csv('{file_path}', 
                  delim='|', 
                  header=TRUE, 
                  types={{'EmergencyAdmitDTS': 'VARCHAR'}}, 
                  ignore_errors=TRUE)
        )
        SELECT * FROM data;
    """)

def create_static_data_vocab(conn: duckdb.DuckDBPyConnection):
    conn.sql("""
        INSERT INTO static_data_vocab (type, description, code, encounters)
        SELECT 'service', COALESCE(max(service_desc), 'Unknown'), COALESCE(service_id::text,'0') as service_id, count(*) as qty
        FROM raw_inpatient
        GROUP BY service_id
        ORDER BY service_id;
        
        INSERT INTO static_data_vocab (type, description, code, encounters)
        SELECT 'disposition', COALESCE(disposition_desc, 'Unknown'), COALESCE(disposition_desc,'Unknown'), count(*) as qty
        FROM raw_inpatient
        GROUP BY disposition_desc
        ORDER BY disposition_desc;
        
        INSERT INTO static_data_vocab (type, description, code, encounters)
        SELECT 'means_of_arrival', COALESCE(max(ma_desc), 'Unknown'), COALESCE(ma_id::text,'0'), count(*) as qty
        FROM raw_inpatient
        GROUP BY ma_id
        ORDER BY ma_id;
        
        INSERT INTO static_data_vocab (type, description, code, encounters)
        SELECT 'race', COALESCE(max(race), 'Unknown'), COALESCE(race::text,'Unknown'), count(*) as qty
        FROM raw_inpatient
        GROUP BY race
        ORDER BY race;
        
        -- binary features
        WITH data AS (
            SELECT feature_name, value, encounter_id
            FROM raw_inpatient
            UNPIVOT (value FOR feature_name IN (is_married, is_speak_english, is_graduated, is_smoked, is_drunk, was_in_icu))
        )
        INSERT INTO static_data_vocab (type, description, code, encounters)
        SELECT feature_name, feature_name, value, count(distinct encounter_id) as qty
        FROM data
        WHERE value = 1
        GROUP BY feature_name, value;
        
        -- numeric features
        WITH data AS (
            SELECT feature_name, value, encounter_id
            FROM raw_inpatient
            UNPIVOT (value FOR feature_name IN (sdi, age, days_from_last))
        ), values AS (
            SELECT feature_name, value, encounter_id
            FROM data
            GROUP BY feature_name, value, encounter_id
        ), feature_groups AS (
            SELECT feature_name, ntile(CASE WHEN feature_name = 'age' THEN 15 ELSE 10 END) OVER (PARTITION BY feature_name ORDER BY value) as vq, value, encounter_id
            FROM values
        )
        INSERT INTO static_data_vocab (type, description, code, lower_value, upper_value,  encounters)
        SELECT 
            feature_name, 
            printf('%d: %s %.1f-%.1f', vq, feature_name, min(value)::float, max(value)::float), 
            vq, min(value), max(value), count(distinct encounter_id)
        FROM feature_groups
        GROUP BY feature_name, vq
        ORDER BY feature_name, vq;
    """)

def create_deduplicated_inpatient_table(conn: duckdb.DuckDBPyConnection):
    conn.sql("""
        DROP TABLE IF EXISTS inpatient;
        CREATE TABLE inpatient AS 
        SELECT * EXCLUDE (prev_admitted, prev_discharged)
        FROM (
            WITH grouped_inpatient AS (
                SELECT 
                    patient_id, encounter_id, 
                    min(admitted) AS admitted, 
                    max(discharged) AS discharged, 
                    LIST(DISTINCT COALESCE(service_id, 0)) AS service_ids,
                    LIST(DISTINCT COALESCE(ma_id,0)) AS ma_ids,
                    max(is_female) AS is_female,
                    max(age) AS age,
                    MIN(COALESCE(race, 'Unknown')) AS race,
                    max(is_married) AS is_married,
                    max(is_speak_english) AS is_speak_english,
                    max(is_graduated) AS is_graduated,
                    COALESCE(avg(sdi), 0) AS sdi,
                    max(is_smoked) AS is_smoked,
                    max(is_drunk) AS is_drunk,
                    max(COALESCE(days_from_last,0)) AS days_from_last,
                    max(in_30) AS in_30,
                    max(was_in_icu) AS was_in_icu
                FROM raw_inpatient
                GROUP BY patient_id, encounter_id
                ORDER BY admitted
            )
            SELECT *,
                lag(admitted) OVER (ORDER BY admitted) AS prev_admitted,
                lag(discharged) OVER (ORDER BY admitted) AS prev_discharged,
                (admitted BETWEEN prev_admitted AND prev_discharged OR 
                discharged BETWEEN prev_admitted AND prev_discharged) AS intersects_prev,
                COUNT(*) OVER (
                    PARTITION BY patient_id 
                    ORDER BY admitted 
                    RANGE BETWEEN unbounded preceding AND CURRENT ROW
                ) AS encounter_num
            FROM grouped_inpatient
        );
    """)

def create_static_feature_array(conn: duckdb.DuckDBPyConnection):
    conn.sql("""
        DROP TABLE IF EXISTS static_features;
        CREATE TABLE static_features AS
        WITH binary_data AS (
            SELECT feature_name, max(value) as value, patient_id, encounter_id
            FROM raw_inpatient
            UNPIVOT (value FOR feature_name IN (is_married, is_speak_english, is_graduated, is_smoked, is_drunk, was_in_icu))
            GROUP BY ALL
        ), binary_features AS (
            SELECT binary_data.patient_id, binary_data.encounter_id, static_data_vocab.id
            FROM binary_data
                INNER JOIN static_data_vocab ON 
                    binary_data.feature_name = static_data_vocab.type AND 
                    binary_data.value::TEXT = static_data_vocab.code
        ), numeric_data AS (
            SELECT feature_name, value, patient_id, encounter_id
            FROM raw_inpatient
            UNPIVOT (value FOR feature_name IN (sdi, age, days_from_last))
        ), numeric_values AS (
            SELECT feature_name, max(value) AS value, patient_id, encounter_id
            FROM numeric_data
            GROUP BY ALL
        ), numeric_features AS (
            SELECT numeric_values.patient_id, numeric_values.encounter_id, static_data_vocab.id
            FROM numeric_values
                ASOF JOIN static_data_vocab 
                    ON 
                        numeric_values.feature_name = static_data_vocab.type AND 
                        numeric_values.value >= static_data_vocab.lower_value
        ), dict_data AS (
            SELECT 
                patient_id, encounter_id, 
                COALESCE(service_id::TEXT, '0') AS service,
                COALESCE(disposition_desc::TEXT, 'Unknown') AS disposition,
                COALESCE(ma_id::TEXT, '0') AS means_of_arrival,
                COALESCE(race, 'Unknown') AS race
            FROM raw_inpatient
            GROUP BY ALL
        ), dict_values AS (
            SELECT *
            FROM dict_data
            UNPIVOT (value FOR feature_name IN (service, disposition, means_of_arrival))
        ), dict_features AS (
            SELECT patient_id, encounter_id, static_data_vocab.id
            FROM dict_values
                    JOIN static_data_vocab 
                        ON 
                            dict_values.feature_name = static_data_vocab.type AND 
                            dict_values.value = static_data_vocab.code
            GROUP BY patient_id, encounter_id, static_data_vocab.id
        )
        SELECT patient_id, encounter_id, LIST(DISTINCT id ORDER BY id) AS features
        FROM (
            FROM binary_features
            UNION ALL
            FROM numeric_features
            UNION ALL
            FROM dict_features
        )
        GROUP BY patient_id, encounter_id;
    """)

def create_encounters_period_table(conn: duckdb.DuckDBPyConnection):
    conn.sql("""
        DROP TABLE IF EXISTS encounters;
        CREATE TABLE encounters AS
        SELECT
            row_number() OVER () AS idx,
            patient_id, encounter_id, 
            min(admitted) as admitted, max(discharged) as discharged
        FROM raw_inpatient
        GROUP BY patient_id, encounter_id        
    """)

def load_labs(conn: duckdb.DuckDBPyConnection, file_path: str):
    conn.sql(f"""
        DROP TABLE IF EXISTS labs_raw;
        CREATE TABLE labs_raw AS
        SELECT 
            StudyID AS patient_id, 
            EncounterID AS encounter_id,
            componentCommonNM AS code,
            NVal AS value,
            HoursSinceAdmit AS hours_since_admit
        FROM read_csv('{file_path}', delim='|', header=TRUE);
    """)


def split_labs_values_into_groups(conn: duckdb.DuckDBPyConnection, labs_limit: float = 1):
    conn.sql(f"""
        DROP TABLE IF EXISTS labs_values;
        CREATE TABLE labs_values AS
        WITH data AS (
            SELECT 
                labs_raw.patient_id,
                labs_raw.encounter_id,
                labs_raw.code,
                labs_raw.value,
                labs_raw.hours_since_admit,
                encounters.admitted,
                encounters.discharged,
                (encounters.admitted + INTERVAL (labs_raw.hours_since_admit::INTEGER) HOUR) AS lab_time,
                extract(hour from encounters.discharged - lab_time) AS hours_to_discharge,
                COUNT(DISTINCT labs_raw.encounter_id) OVER (PARTITION BY code) AS lab_encounters_num,
                COUNT(DISTINCT labs_raw.encounter_id) OVER () AS total_encounters_num
            FROM labs_raw
                INNER JOIN encounters ON 
                    labs_raw.patient_id = encounters.patient_id AND 
                    labs_raw.encounter_id = encounters.encounter_id
            WHERE lab_time BETWEEN encounters.admitted AND encounters.discharged
        )
        SELECT 
            patient_id, 
            encounter_id,
            code,
            value, 
            hours_since_admit,
            hours_to_discharge,
            ntile(10) OVER (PARTITION BY code ORDER BY value) as value_group
        FROM data
        WHERE 
            lab_encounters_num::FLOAT / total_encounters_num::FLOAT >= {labs_limit}
    """)

def create_labs_values_vocab(conn: duckdb.DuckDBPyConnection):
    conn.sql("""
        INSERT INTO dynamic_data_vocab (type, description, code, value_group, lower_value, upper_value, encounters) 
        SELECT 'labs', code, code, value_group, min(value), max(value), COUNT(DISTINCT encounter_id) as qty
        FROM labs_values
        GROUP BY code, value_group
        ORDER BY code, value_group;
    """)


def create_labs_features(conn: duckdb.DuckDBPyConnection, aggregation_hours: int):
    conn.sql(f"""
        DROP TABLE IF EXISTS dynamic_labs_data;
        CREATE TABLE dynamic_labs_data AS
        SELECT patient_id,
               encounter_id,
               FLOOR(hours_since_admit / {aggregation_hours})::INT AS period,
               min(dynamic_data_vocab.id) AS value
        FROM labs_values
                INNER JOIN dynamic_data_vocab ON 
                    dynamic_data_vocab.type = 'labs' AND
                    labs_values.code = dynamic_data_vocab.code AND 
                    labs_values.value_group = dynamic_data_vocab.value_group
        GROUP BY patient_id, encounter_id, period, labs_values.code;
    """)

def load_vitals(conn: duckdb.DuckDBPyConnection, vitals_paths: dict):
    conn.sql("DROP TABLE IF EXISTS vitals_raw;")
    conn.sql("""
        CREATE TABLE vitals_raw (
            patient_id VARCHAR,
            encounter_id BIGINT,
            type VARCHAR,
            code VARCHAR,
            value FLOAT,
            hours_since_admit FLOAT
        );
    """)

    if vitals_paths.get("vitals_o2_device_file") and vitals_paths.get("o2_device_mapping_file"):
        conn.sql(f"""
            INSERT INTO vitals_raw BY NAME
            SELECT 
                StudyID AS patient_id, 
                EncounterID AS encounter_id,
                'O2Device' AS type,
                mapping.category AS code,
                0 AS value,
                HoursSinceAdmit AS hours_since_admit
            FROM read_csv('{vitals_paths["vitals_o2_device_file"]}') AS data
            INNER JOIN read_csv('{vitals_paths["o2_device_mapping_file"]}') AS mapping
            ON data.MeasureTXT = mapping.description;
        """)

    for name, code in [
        ("vitals_dbp_file", "DBP"),
        ("vitals_pulse_file", "Pulse"),
        ("vitals_resp_rate_file", "RespRate"),
        ("vitals_sbp_file", "SBP"),
        ("vitals_sp_o2_file", "SpO2"),
        ("vitals_temp_file", "Temp"),
    ]:
        if vitals_paths.get(name):
            value_expr = "try_cast(MeasureTXT AS FLOAT)" if code == "SpO2" else "MeasureTXT"
            conn.sql(f"""
                INSERT INTO vitals_raw BY NAME
                SELECT 
                    StudyID AS patient_id, 
                    EncounterID AS encounter_id,
                    '{code}' AS type,
                    '{code}' AS code,
                    {value_expr} AS value,
                    HoursSinceAdmit AS hours_since_admit
                FROM read_csv('{vitals_paths[name]}');
            """)

    conn.sql("DROP TABLE IF EXISTS vitals_values;")
    conn.sql("""
        CREATE TABLE vitals_values AS
        SELECT 
            vitals_raw.patient_id,
            vitals_raw.encounter_id, 
            type,
            code, 
            ntile(CASE WHEN type = 'O2Device' THEN 1 ELSE 10 END) OVER (PARTITION BY type, code ORDER BY value) as value_group,
            hours_since_admit,
            (encounters.admitted + INTERVAL (vitals_raw.hours_since_admit::INTEGER) HOUR) AS measure_time,
            extract(hour from encounters.discharged - measure_time) AS hours_to_discharge,
            value
        FROM vitals_raw
            INNER JOIN encounters ON vitals_raw.patient_id = encounters.patient_id AND vitals_raw.encounter_id = encounters.encounter_id
        WHERE measure_time BETWEEN encounters.admitted AND encounters.discharged
    """)

def create_vitals_values_vocab(conn: duckdb.DuckDBPyConnection):
    conn.sql("""
        INSERT INTO dynamic_data_vocab (type, description, code, value_group, lower_value, upper_value, encounters) 
        SELECT type, code, code, value_group, min(value), max(value), COUNT(DISTINCT encounter_id) as qty
        FROM vitals_values
        GROUP BY type, code, value_group
        ORDER BY type, code, value_group;
    """)


def create_dynamic_vitals_features(conn: duckdb.DuckDBPyConnection, aggregation_hours: int):
    conn.sql(f"""
        DROP TABLE IF EXISTS dynamic_vitals_data;
        CREATE TABLE dynamic_vitals_data AS
        SELECT patient_id,
               encounter_id,
               FLOOR(hours_since_admit / {aggregation_hours})::INT AS period,
               min(dynamic_data_vocab.id) AS value
        FROM vitals_values
                INNER JOIN dynamic_data_vocab ON 
                    vitals_values.type = dynamic_data_vocab.type AND
                    vitals_values.code = dynamic_data_vocab.code AND 
                    vitals_values.value_group = dynamic_data_vocab.value_group
        GROUP BY patient_id, encounter_id, period, vitals_values.code;
    """)

def load_diagnoses(conn: duckdb.DuckDBPyConnection, diagnoses_file: str, data_path: pathlib.Path):
    conn.sql(f"""
        DROP TABLE IF EXISTS diagnoses_raw;
        CREATE TABLE diagnoses_raw AS
        SELECT 
            StudyID AS patient_id,
            EncounterID AS encounter_id,
            HoursSinceAdmit AS hours_since_admit,
            ICD10CD AS code
        FROM read_csv('{diagnoses_file}');
        
        DROP TABLE IF EXISTS codes;
        CREATE TABLE codes AS
        FROM '{data_path}/codes.parquet';
        
        DROP TABLE IF EXISTS diagnoses_values;
        CREATE TABLE diagnoses_values AS
        SELECT 
            diagnoses_raw.patient_id, diagnoses_raw.encounter_id, code, hours_since_admit,
            (encounters.admitted + INTERVAL (diagnoses_raw.hours_since_admit::INTEGER) HOUR) AS diagnoses_time,
            extract(hour from encounters.discharged - diagnoses_time) AS hours_to_discharge
        FROM diagnoses_raw
            INNER JOIN encounters ON 
                diagnoses_raw.patient_id = encounters.patient_id AND 
                diagnoses_raw.encounter_id = encounters.encounter_id
        WHERE diagnoses_time BETWEEN encounters.admitted AND encounters.discharged;
    """)

def create_diagnoses_values_vocab(conn: duckdb.DuckDBPyConnection, diagnoses_limit: float = 1):
    conn.sql(f"""
        WITH data AS (
            SELECT 
                diagnoses_values.code,COALESCE(codes.description, '') as description, 
                COUNT(DISTINCT encounter_id) as encounters,
                COUNT(DISTINCT patient_id) as patients
            FROM diagnoses_values
                LEFT JOIN codes ON 
                    codes.type = 'ICD10' AND 
                    diagnoses_values.code = codes.code
            GROUP BY diagnoses_values.code, description
        ), total_encounters AS (
            SELECT COUNT(DISTINCT patient_id) as qty
            FROM encounters
        )
        INSERT INTO dynamic_data_vocab (type, description, code, value_group, lower_value, upper_value, encounters) 
        SELECT 'diagnoses', description, code, 1, 0, 0, encounters
        FROM data
        WHERE patients::FLOAT / (SELECT qty FROM total_encounters LIMIT 1) >= {diagnoses_limit}; 
    """)

def create_diagnoses_features(conn: duckdb.DuckDBPyConnection, aggregation_hours: int):
    conn.sql(f"""
        DROP TABLE IF EXISTS dynamic_diagnoses_data;
        CREATE TABLE dynamic_diagnoses_data AS
        SELECT patient_id,
               encounter_id,
               FLOOR(hours_since_admit / {aggregation_hours})::INT AS period,
               min(dynamic_data_vocab.id) AS value
        FROM diagnoses_values
                INNER JOIN dynamic_data_vocab ON
                    dynamic_data_vocab.type = 'diagnoses' AND
                    diagnoses_values.code = dynamic_data_vocab.code
        GROUP BY patient_id, encounter_id, period, diagnoses_values.code;
    """)

def load_procedures(conn: duckdb.DuckDBPyConnection, procedures_file: str):
    conn.sql(f"""
        DROP TABLE IF EXISTS procedures_raw;
        CREATE TABLE procedures_raw AS
        SELECT 
            EncounterID AS encounter_id,
            HoursSinceAdmit_order AS hours_since_admit,
            ProcedureCD AS code,
            ANY_VALUE(ProcedureDSC) AS description
        FROM read_csv('{procedures_file}')
        GROUP BY EncounterID, HoursSinceAdmit_order, ProcedureCD;
        
        DROP TABLE IF EXISTS procedures_values;
        CREATE TABLE procedures_values AS
        SELECT 
            encounters.patient_id, procedures_raw.encounter_id, code, description, hours_since_admit,
            (encounters.admitted + INTERVAL (procedures_raw.hours_since_admit::INTEGER) HOUR) AS procedure_time,
            extract(hour from encounters.discharged - procedure_time) AS hours_to_discharge
        FROM procedures_raw
            INNER JOIN encounters ON 
                procedures_raw.encounter_id = encounters.encounter_id
        WHERE procedure_time BETWEEN encounters.admitted AND encounters.discharged;
    """)

def create_procedures_values_vocab(conn: duckdb.DuckDBPyConnection, procedures_limit: float = 1):
    conn.sql(f"""
        WITH data AS (
            SELECT 
                procedures_values.code,
                COALESCE(procedures_values.description, '') as description, 
                COUNT(DISTINCT encounter_id) as encounters,
                COUNT(DISTINCT patient_id) as patients
            FROM procedures_values
            GROUP BY code, description
        ), total_encounters AS (
            SELECT COUNT(DISTINCT patient_id) as qty
            FROM encounters
        )
        INSERT INTO dynamic_data_vocab (type, description, code, value_group, lower_value, upper_value, encounters) 
        SELECT 'procedures', description, code, 1, 0, 0, encounters
        FROM data
        WHERE patients::FLOAT / (SELECT qty FROM total_encounters LIMIT 1) >= {procedures_limit};
    """)

def create_procedures_features(conn: duckdb.DuckDBPyConnection, aggregation_hours: int):
    conn.sql(f"""
        DROP TABLE IF EXISTS dynamic_procedures_data;
        CREATE TABLE dynamic_procedures_data AS
        SELECT patient_id,
               encounter_id,
               FLOOR(hours_since_admit / {aggregation_hours})::INT AS period,
               min(dynamic_data_vocab.id) AS value
        FROM procedures_values
                INNER JOIN dynamic_data_vocab ON
                    dynamic_data_vocab.type = 'procedures' AND
                    procedures_values.code = dynamic_data_vocab.code
        GROUP BY patient_id, encounter_id, period, procedures_values.code;
    """)

def load_medications(conn: duckdb.DuckDBPyConnection, medications_file: str):
    conn.sql(f"""
        DROP TABLE IF EXISTS medications_raw;
        CREATE TABLE medications_raw AS
        SELECT
            StudyID AS patient_id,
            EncounterID AS encounter_id,
            HoursSinceAdmit AS hours_since_admit,
            PharmaceuticalSubclassCD AS code,
            ANY_VALUE(PharmaceuticalSubclassDSC) AS description
        FROM read_csv('{medications_file}')
        GROUP BY StudyID, EncounterID, HoursSinceAdmit, PharmaceuticalSubclassCD;
        
        DROP TABLE IF EXISTS medications_values;
        CREATE TABLE medications_values AS
        SELECT 
            encounters.patient_id, medications_raw.encounter_id, code, description, hours_since_admit,
            (encounters.admitted + INTERVAL (medications_raw.hours_since_admit::INTEGER) HOUR) AS procedure_time,
            extract(hour from encounters.discharged - procedure_time) AS hours_to_discharge
        FROM medications_raw
            INNER JOIN encounters ON
                medications_raw.patient_id = encounters.patient_id AND
                medications_raw.encounter_id = encounters.encounter_id
        WHERE procedure_time BETWEEN encounters.admitted AND encounters.discharged;
    """)

def create_medications_values_vocab(conn: duckdb.DuckDBPyConnection, medications_limit: float = 1):
    conn.sql(f"""
        WITH data AS (
            SELECT 
                medications_values.code,
                COALESCE(medications_values.description, '') as description, 
                COUNT(DISTINCT encounter_id) as encounters,
                COUNT(DISTINCT patient_id) as patients
            FROM medications_values
            GROUP BY code, description
        ), total_encounters AS (
            SELECT COUNT(DISTINCT patient_id) as qty
            FROM encounters
        )
        INSERT INTO dynamic_data_vocab (type, description, code, value_group, lower_value, upper_value, encounters) 
        SELECT 'medications', description, code, 1, 0, 0, encounters
        FROM data
        WHERE patients::FLOAT / (SELECT qty FROM total_encounters LIMIT 1) >= {medications_limit}; 
    """)

def create_medications_features(conn: duckdb.DuckDBPyConnection, aggregation_hours: int):
    conn.sql(f"""
        DROP TABLE IF EXISTS dynamic_medications_data;
        CREATE TABLE dynamic_medications_data AS
        SELECT patient_id,
               encounter_id,
               FLOOR(hours_since_admit / {aggregation_hours})::INT AS period,
               min(dynamic_data_vocab.id) AS value
        FROM medications_values
                INNER JOIN dynamic_data_vocab ON
                    dynamic_data_vocab.type = 'medications' AND
                    medications_values.code = dynamic_data_vocab.code
        GROUP BY patient_id, encounter_id, period, medications_values.code;
    """)

def create_dynamic_features_arrays(conn: duckdb.DuckDBPyConnection):
    conn.sql("""
        DROP TABLE IF EXISTS dynamic_features;
        CREATE TABLE dynamic_features AS
        WITH data AS (
            SELECT patient_id, encounter_id, period, value 
            FROM dynamic_labs_data
            UNION ALL
            SELECT patient_id, encounter_id, period, value
            FROM dynamic_vitals_data
            UNION ALL
            SELECT patient_id, encounter_id, period, value
            FROM dynamic_diagnoses_data
            UNION ALL
            SELECT patient_id, encounter_id, period, value
            FROM dynamic_procedures_data
            UNION ALL
            SELECT patient_id, encounter_id, period, value
            FROM dynamic_medications_data
        ), hourly_data AS (
            SELECT patient_id, encounter_id, period, LIST(DISTINCT value ORDER BY value) AS features
            FROM data
            GROUP BY patient_id, encounter_id, period
        )
        SELECT patient_id, encounter_id, LIST(features ORDER BY period) AS features, LIST(DISTINCT period ORDER BY period) AS periods
        FROM hourly_data
        GROUP BY patient_id, encounter_id
    """)

def save_static_data_vocab(conn: duckdb.DuckDBPyConnection, output_path: pathlib.Path):
    conn.query(
        f"COPY static_data_vocab TO '{output_path / 'static_data_vocab.parquet'}' (FORMAT PARQUET);")

def save_dynamic_data_vocab(conn: duckdb.DuckDBPyConnection, output_path: pathlib.Path):
    conn.query(
        f"COPY dynamic_data_vocab TO '{output_path / 'dynamic_data_vocab.parquet'}' (FORMAT PARQUET);")

def save_data(conn: duckdb.DuckDBPyConnection, aggregation_hours: int, output_path: pathlib.Path):
    conn.sql(f"""
        DROP TABLE IF EXISTS data;
        CREATE TABLE data AS
        SELECT
            encounters.idx as id,
            row_number() OVER () AS idx,
            static_features.features AS static_features,
            dynamic_features.features AS dynamic_features,
            dynamic_features.periods AS periods,
            FLOOR((hour(encounters.discharged - encounters.admitted) +
            day(encounters.discharged - encounters.admitted)*24) / {aggregation_hours})::INT AS duration
        FROM encounters
                INNER JOIN static_features ON 
                    encounters.patient_id = static_features.patient_id AND 
                    encounters.encounter_id = static_features.encounter_id
                INNER JOIN dynamic_features ON
                    encounters.patient_id = dynamic_features.patient_id AND 
                    encounters.encounter_id = dynamic_features.encounter_id
        ORDER BY encounters.idx;
    """)
    conn.sql(f"""
        COPY (
            SELECT
                data.idx,
                static_features,
                dynamic_features,
                periods,
                duration
            FROM data
        ) TO '{output_path / 'data.parquet'}' (FORMAT parquet);
    """)

def save_target(conn: duckdb.DuckDBPyConnection, output_path: pathlib.Path):
    conn.query(f"""
        COPY (
            SELECT 
                data.idx,
                inpatient.in_30 AS target
            FROM encounters
                INNER JOIN inpatient ON 
                    inpatient.patient_id = encounters.patient_id AND 
                    inpatient.encounter_id = encounters.encounter_id
                INNER JOIN data ON 
                        encounters.idx = data.id
        ) TO '{output_path / 'target.parquet'}' (FORMAT parquet);
    """)

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser(description='Pre-process data for the dataset for the diabetes A1C1 level prediction task')
    parser.add_argument('--config', required=True, type=str, help='path to the configuration file')
    args = parser.parse_args()
    if args.config == ""  or not os.path.exists(args.config):
        print('Configuration file does not exist')
        sys.exit(1)

    configs = load_dataset_config(args.config)
    for idx, config in enumerate(configs):
        print(f"🔧 Processing dataset {idx + 1}/{len(configs)} — output: {config.output_path}")
        process_data(config)

    end = time.time()
    duration = end - start
    minutes, seconds = divmod(duration, 60)
    print(f"⏱️ Script finished in {int(minutes)} min {seconds:.1f} sec")
