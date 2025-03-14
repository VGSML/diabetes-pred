import duckdb
import pandas as pd
import pathlib
import yaml

class Config:
    def __init__(self, outcomesFile: str = 'diabetes_pred/data/DiabetesOutcomes.txt',
                 diagnosesFile: str = 'diabetes_pred/data/DiagnosesICD10.txt',
                 proceduresFile: str = 'diabetes_pred/data/ProceduresICD10.txt',
                 labsFile: str = 'diabetes_pred/data/Labs_numeric_lim_unicode.txt',
                 diagnosesLimit: float = 0.04,
                 proceduresLimit: float = 0.05,
                 labsLimit: float = 0.05,
                 useNonNumericLabs: bool = False,
                 quantizeLabs: bool = False,
                 months: int = 36,
                 dbPath: str = 'diabetes_pred/data/process.duckdb',
                 outputPath: str = 'diabetes_pred/data/processed/'):
        self.outcomesFile = outcomesFile
        self.diagnosesFile = diagnosesFile
        self.proceduresFile = proceduresFile
        self.labsFile = labsFile
        self.diagnosesLimit = diagnosesLimit
        self.proceduresLimit = proceduresLimit
        self.labsLimit = labsLimit
        self.useNonNumericLabs = useNonNumericLabs
        self.quantizeLabs = quantizeLabs
        self.months = months
        self.dbPath = dbPath
        self.outputPath = outputPath

def load_dataset_config(path: str):
    cc = []

    with open(path, 'r') as f:
        conf = yaml.safe_load(f)

        #TODO: test code below.
        #can i replace all with this code?
        # return [Config(**c) for c in conf['datasets']]

        for c in conf['datasets']:
            cc.append( Config(
                outcomesFile=c['outcomesFile'],
                diagnosesFile=c['diagnosesFile'],
                proceduresFile=c['proceduresFile'],
                labsFile=c.get('labsFile'),
                diagnosesLimit=c['diagnosesLimit'],
                proceduresLimit=c['proceduresLimit'],
                labsLimit=c.get('labsLimit'),
                useNonNumericLabs=c['useNonNumericLabs'],
                quantizeLabs=c['quantizeLabs'],
                months=c['months'],
                dbPath=c.get('dbPath'),
                outputPath=c['outputPath']
            ))

    return cc

def process_config(path: str):
    cc = load_dataset_config(path)

    for c in cc:
        process_data(c)

def process_data(conf: Config):
    print('Processing data for:', vars(conf))
    dbPath = conf.dbPath if conf.dbPath else ':memory:'
    conn = duckdb.connect(dbPath)
    createSchema(conn)

    #number of patients
    patients = fileRows(conn, conf.outcomesFile)

    #load diagnoses, procedures, laboratory tests
    loadDynamicData(conn, conf.diagnosesFile,'diagnoses', conf.months, patients*conf.diagnosesLimit)
    loadDynamicData(conn, conf.proceduresFile, 'procedures', conf.months, patients*conf.proceduresLimit)
    use_labs = True if conf.labsFile else False
    if use_labs:
        loadLabs(conn, conf.labsFile, conf.months, patients*conf.labsLimit)

    #Loading static data
    loadStaticData(conn, conf.outcomesFile, use_labs=use_labs)

    #Creating feature dictionaries
    makeStaticDataVocab(conn)
    makeDynamicDataVocab(conn, 'diagnoses', 'code_with_type')
    makeDynamicDataVocab(conn, 'procedures', 'code_with_type')
    if use_labs:
        makeLabsValuesVocab(conn, conf.useNonNumericLabs, conf.quantizeLabs)

    #Creating the final table
    makeDataTable(conn, conf.months)

    # save the processed data
    pathlib.Path(conf.outputPath).mkdir(parents=True, exist_ok=True)

    # save vocabs
    conn.query(f"COPY static_data_vocab TO '{pathlib.Path(conf.outputPath) / 'static_data_vocab.parquet'}' (FORMAT PARQUET);")
    conn.query(f"COPY dynamic_data_feature_vocab TO '{pathlib.Path(conf.outputPath) / 'dynamic_data_vocab.parquet'}' (FORMAT PARQUET);")

    # save the processed data
    conn.query(f"""
        COPY (SELECT * FROM data ORDER BY idx)
        TO '{pathlib.Path(conf.outputPath) / 'data.parquet'}' (FORMAT PARQUET);
    """)
    conn.query(f"""
        COPY (
            SELECT data.idx, subjects.id, outcomes.a1_greater_7
            FROM subjects
                INNER JOIN outcomes ON subjects.study_id = outcomes.study_id
                INNER JOIN data ON subjects.id = data.id
            ORDER BY data.idx
        ) TO '{pathlib.Path(conf.outputPath) / 'target.parquet'}' (FORMAT PARQUET);
    """)

    conn.execute("VACUUM")
    conn.close()


def fileRows(conn: duckdb.DuckDBPyConnection, file_name: str):
    return conn.sql(f"SELECT Count(*) FROM read_csv('{file_name}', delim='|', header=TRUE)").fetchone()[0]

def createSchema(conn: duckdb.DuckDBPyConnection):
    # subjects
    conn.query(f"DROP SEQUENCE IF EXISTS subjects_id_seq CASCADE;")
    conn.query(f"CREATE SEQUENCE subjects_id_seq;")
    conn.query(f"DROP TABLE IF EXISTS subjects;")
    conn.query(f"""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY DEFAULT nextval('subjects_id_seq'),
            study_id VARCHAR NOT NULL,
            UNIQUE(study_id)
        );
    """)

    # static data vocab
    conn.query(f"DROP SEQUENCE IF EXISTS static_data_vocab_id_seq CASCADE;")
    conn.query(f"CREATE SEQUENCE static_data_vocab_id_seq;")
    conn.query(f"DROP TABLE IF EXISTS static_data_vocab;")
    conn.query(f"""
        CREATE TABLE IF NOT EXISTS static_data_vocab (
            id INTEGER PRIMARY KEY DEFAULT nextval('static_data_vocab_id_seq'),
            type VARCHAR NOT NULL,
            value VARCHAR NOT NULL,
            lower_bound FLOAT NOT NULL DEFAULT 0,
            upper_bound FLOAT NOT NULL DEFAULT 0,
            subjects_count INTEGER DEFAULT 0,
            UNIQUE(type, value)
        );
    """)
    # dynamic data vocab
    conn.query(f"DROP SEQUENCE IF EXISTS dynamic_data_feature_vocab_id_seq CASCADE;")
    conn.query(f"CREATE SEQUENCE dynamic_data_feature_vocab_id_seq;")
    conn.query(f"DROP TABLE IF EXISTS dynamic_data_feature_vocab;")
    conn.query(f"""
        CREATE TABLE IF NOT EXISTS dynamic_data_feature_vocab (
            id INTEGER PRIMARY KEY DEFAULT nextval('dynamic_data_feature_vocab_id_seq'),
            type VARCHAR NOT NULL,
            code VARCHAR NOT NULL,
            text_res VARCHAR NOT NULL DEFAULT '',
            lower_bound FLOAT NOT NULL DEFAULT 0,
            upper_bound FLOAT NOT NULL DEFAULT 0,
            avg_value FLOAT NOT NULL DEFAULT 0,
            stddev FLOAT NOT NULL DEFAULT 0,
            event_count INTEGER DEFAULT 0,
            subjects_count INTEGER DEFAULT 0,
            UNIQUE(type, code, text_res)
        );
    """)

def loadStaticData(conn: duckdb.DuckDBPyConnection, file_path: str, use_labs=False):
    conn.query(f"DROP TABLE IF EXISTS outcomes;")
    if use_labs:
        conn.query(f"""
            INSERT INTO subjects (study_id)
            SELECT DISTINCT study_id
            FROM diagnoses
                INNER JOIN procedures USING (study_id)
                INNER JOIN labs USING (study_id)
        """)
    else:
        conn.query(f"""
            INSERT INTO subjects (study_id)
            SELECT DISTINCT study_id
            FROM diagnoses
                    INNER JOIN procedures USING (study_id)
        """)
    conn.query(f"""
        CREATE TABLE outcomes AS
        SELECT
            data.StudyID AS study_id,
            data.IndexDate::TIMESTAMP AS index_date,
            data.InitialA1c AS initial_a1c,
            data.A1cGreaterThan7 AS a1_greater_7,
            data.Female AS female,
            data.Married AS married,
            data.GovIns AS gov_ins,
            data.English AS english,
            data.AgeYears AS age_years, 
            data.SDI_score AS sdi_score,
            data.Veteran AS veteran,
            data.DaysFromIndexToInitialA1cDate AS days_to_initial_a1c_date,
            data.DaysFromIndexToA1cDateAfter12Months AS days_to_a1c_after_12_months, 
            data.DaysFromIndexToFirstEncounterDate AS days_to_first_encounter_date,
            data.DaysFromIndexToLastEncounterDate AS days_to_last_encounter_date,
            data.DaysFromIndexToLatestDate AS days_to_latest_date,
            data.DaysFromIndexToPatientTurns18 AS days_to_patient_turns_18
        FROM read_csv('{file_path}', delim='|', header=TRUE) AS data
            INNER JOIN subjects ON data.StudyID = subjects.study_id;
    """)

def makeStaticDataVocab(conn: duckdb.DuckDBPyConnection):
    conn.sql("""
        WITH data AS (
            SELECT study_id, index_date, column_name, value, 1 AS subject_count
            FROM outcomes
            UNPIVOT(value FOR column_name IN (initial_a1c, a1_greater_7, female, married, gov_ins, english, age_years, sdi_score, veteran))
        ), binary_features AS (
            SELECT column_name, value, SUM(subject_count) AS subject_count
            FROM data
            WHERE column_name IN ('female', 'married', 'gov_ins', 'english', 'veteran')
            GROUP BY column_name, value
            ORDER BY column_name, value
        ), numeric_values AS (
            SELECT
                column_name,
                ntile(10) OVER (PARTITION BY column_name ORDER BY value) AS value_quantile,
                value,
                subject_count
            FROM data
            WHERE column_name IN ('initial_a1c', 'age_years', 'sdi_score')
        ), numeric_features AS (
            SELECT 
                column_name,
                min(value) AS lower_bound,
                max(value) AS upper_bound,
                printf('%d %.1f-%.1f', value_quantile, lower_bound::float, upper_bound::float) AS value,
                SUM(subject_count) AS subject_count
            FROM numeric_values
            GROUP BY column_name, value_quantile
            ORDER BY column_name, value_quantile
        ), union_data AS (
            SELECT column_name, value, lower_bound, upper_bound, subject_count
            FROM numeric_features
            UNION
            SELECT column_name, value::VARCHAR, value, value, subject_count
            FROM binary_features
        )
        INSERT INTO static_data_vocab (type, value, lower_bound, upper_bound, subjects_count)
        SELECT column_name, value, lower_bound, upper_bound, subject_count
        FROM union_data
        ORDER BY column_name, lower_bound;
    """)

def loadDynamicData(conn: duckdb.DuckDBPyConnection, file_path: str, table_name: str, limit_months=36, limit_count=0):
    conn.query(f"DROP TABLE IF EXISTS {table_name};")
    conn.query(f"""
        CREATE TABLE {table_name} AS
        WITH data AS (
            SELECT 
                StudyID AS study_id, 
                Date::TIMESTAMP AS event_date, 
                (
                    extract('month' from IndexDate::TIMESTAMP) - extract('month' from Date::TIMESTAMP) +
                    12*(extract('year' from IndexDate::TIMESTAMP) - extract('year' from Date::TIMESTAMP))
                )::INTEGER AS month, 
                Code AS code, 
                Code_Type AS code_type, 
                IndexDate::TIMESTAMP AS index_date, 
                CodeWithType as code_with_type,
                COUNT(DISTINCT study_id) OVER (PARTITION BY code_with_type) AS subject_count
            FROM read_csv('{file_path}', delim='|', header=TRUE) 
            WHERE 
                (IndexDate::TIMESTAMP - Date::TIMESTAMP BETWEEN INTERVAL '1 month' AND INTERVAL '{limit_months} months' OR {limit_months} = 0)
        )
        SELECT * 
        FROM data
        WHERE (subject_count >= {limit_count} OR {limit_count} = 0);
    """)

def makeDynamicDataVocab(conn: duckdb.DuckDBPyConnection, table_name: str, field_name: str):
    conn.query(f"""
        INSERT INTO dynamic_data_feature_vocab (type, code, event_count, subjects_count)
        SELECT '{table_name}', {field_name}, COUNT(*) AS event_count, COUNT(DISTINCT study_id) AS subjects_count
        FROM {table_name}
        GROUP BY {field_name}
        ON CONFLICT(type, code, text_res) DO 
            UPDATE 
                SET 
                    event_count = event_count + excluded.event_count, 
                    subjects_count = subjects_count + excluded.subjects_count;
    """)

def loadLabs(conn: duckdb.DuckDBPyConnection, file_path: str, limit_months=36, limit_count=0, use_df=False):
    conn.query(f"DROP TABLE IF EXISTS labs;")
    df = None
    if use_df:
        df = pd.read_csv('diabetes_pred/data/Labs.txt', sep='|', encoding='cp1252')
        
    conn.query(f"""
        CREATE TABLE labs AS
        WITH subjects AS (
            SELECT DISTINCT study_id, diagnoses.index_date
            FROM diagnoses
                INNER JOIN procedures USING (study_id)
        ), data AS (       
            SELECT 
                subjects.study_id,
                subjects.index_date,
                (
                    extract('month' from subjects.index_date) - extract('month' from data.Date::TIMESTAMP) +
                    12*(extract('year' from subjects.index_date) - extract('year' from data.Date::TIMESTAMP))
                )::INTEGER AS month,
                data.Date::TIMESTAMP AS event_date,
                data.Code AS code,
                data.Result AS result,
                TRY_CAST(data.Result AS FLOAT) AS num_res,
                data.Source AS source,
                COUNT(DISTINCT study_id) OVER (PARTITION BY code, source) AS subject_count
            FROM {"df" if use_df else f"read_csv('{file_path}', delim='|', header=TRUE, quote='', ignore_errors=true)"} AS data
                    INNER JOIN subjects ON data.StudyID = subjects.study_id
            WHERE 
                (subjects.index_date - event_date BETWEEN INTERVAL '1 month' AND INTERVAL '{limit_months} months' OR {limit_months} = 0)
        )
        SELECT
            data.study_id,
            data.index_date,
            data.month,
            max(data.event_date) AS event_date,
            data.code,
            data.source,
            last(data.result ORDER BY data.event_date) AS result,
            avg(data.num_res) AS num_res
        FROM data
        WHERE (subject_count > {limit_count} OR {limit_count} = 0) AND num_res IS NULL
        GROUP BY data.study_id, data.index_date, data.month, data.code, data.source
        UNION
        SELECT
            data.study_id,
            data.index_date,
            data.month,
            max(data.event_date) AS event_date,
            data.code,
            data.source,
            avg(data.num_res)::VARCHAR AS result,
            avg(data.num_res) AS num_res
        FROM data
        WHERE (subject_count > {limit_count} OR {limit_count} = 0) AND num_res IS NOT NULL
        GROUP BY data.study_id, data.index_date, data.month, data.code, data.source;
    """)

def makeLabsValuesVocab(conn: duckdb.DuckDBPyConnection, add_non_num=True, quantize=False):
    if add_non_num:
        conn.query(f"""
            INSERT INTO dynamic_data_feature_vocab (type, code, text_res, event_count, subjects_count)
            SELECT 
                'labs', 
                code, 
                result, 
                COUNT(*) AS event_count, 
                COUNT(DISTINCT study_id) AS subjects_count
            FROM labs
            WHERE num_res IS NULL AND result IS NOT NULL
            GROUP BY code, result
            ON CONFLICT(type, code, text_res) DO 
                UPDATE SET 
                    event_count = event_count + excluded.event_count, 
                    subjects_count = subjects_count + excluded.subjects_count;
    """)
    if not quantize:
        conn.query(f"""
            INSERT INTO dynamic_data_feature_vocab (type, code, text_res, lower_bound, upper_bound, avg_value, stddev, event_count, subjects_count)
            SELECT 
                'labs', 
                code, 
                '', 
                min(num_res) AS lower_bound, 
                max(num_res) AS upper_bound,
                avg(num_res) AS avg_value,
                stddev(num_res) AS stddev,
                COUNT(*) AS event_count, 
                COUNT(DISTINCT study_id) AS subjects_count
            FROM labs
            WHERE num_res IS NOT NULL AND 
                code NOT IN (
                    SELECT code FROM labs
                    WHERE num_res IS NULL AND result IS NOT NULL
                )
            GROUP BY code
            ON CONFLICT(type, code, text_res) DO 
                UPDATE 
                    SET 
                        event_count = event_count + excluded.event_count, 
                        subjects_count = subjects_count + excluded.subjects_count;
        """)
        return 

    conn.query(f"""
        WITH data AS (
            SELECT 
                study_id,
                code,
                num_res AS value,
                ntile(10) OVER (PARTITION BY code ORDER BY num_res) AS value_quantile
            FROM labs
            WHERE num_res IS NOT NULL AND 
                code NOT IN (
                    SELECT code FROM labs
                    WHERE num_res IS NULL AND result IS NOT NULL
                )
        )
        INSERT INTO dynamic_data_feature_vocab (type, code, text_res, lower_bound, upper_bound, event_count, subjects_count)
        SELECT 
            'labs',
            code, 
            printf('%d %f - %f', value_quantile, min(value), max(value)) AS text_res,
            min(value) AS lower_bound, 
            max(value) AS upper_bound, 
            COUNT(*) AS event_count,
            COUNT(DISTINCT study_id) AS subjects_count
        FROM data
        GROUP BY code, value_quantile
        ORDER BY code, value_quantile
        ON CONFLICT(type, code, text_res) DO 
            UPDATE 
                SET 
                    event_count = event_count + excluded.event_count, 
                    subjects_count = subjects_count + excluded.subjects_count;
    """)

def makeDataTable(conn: duckdb.DuckDBPyConnection, months=36):
    conn.query(f"DROP TABLE IF EXISTS data;")
    conn.query(f"DROP SEQUENCE IF EXISTS data_idx_seq CASCADE;")
    conn.query(f"CREATE SEQUENCE data_idx_seq;")
    conn.query(f"""
        CREATE TABLE data AS
        WITH static_features_values AS (
            SELECT study_id, column_name, value
            FROM outcomes
            UNPIVOT(value FOR column_name IN (initial_a1c, female, married, gov_ins, english, age_years, sdi_score, veteran))
        ), static AS (
            SELECT study_id, array_agg(static_data_vocab.id ORDER BY id) as data
            FROM static_features_values
                ASOF JOIN static_data_vocab ON 
                    static_features_values.column_name = static_data_vocab.type AND 
                    static_features_values.value >= static_data_vocab.lower_bound
            GROUP BY study_id
        ), diag AS (
            SELECT 
                study_id,
                month,
                array_agg(dynamic_data_feature_vocab.id ORDER BY id) AS data,
                array_agg(1 ORDER BY id) AS values
            FROM diagnoses
                INNER JOIN dynamic_data_feature_vocab ON diagnoses.code_with_type = dynamic_data_feature_vocab.code and dynamic_data_feature_vocab.type = 'diagnoses'
            GROUP BY study_id, month
        ), proc AS (
            SELECT 
                study_id,
                month,
                array_agg(dynamic_data_feature_vocab.id ORDER BY id) AS data,
                array_agg(1 ORDER BY id) AS values
            FROM procedures
                INNER JOIN dynamic_data_feature_vocab ON procedures.code_with_type = dynamic_data_feature_vocab.code and dynamic_data_feature_vocab.type = 'procedures'
            GROUP BY study_id, month
        ), lab_binary AS (
            SELECT 
                study_id,
                month,
                array_agg(dynamic_data_feature_vocab.id ORDER BY id) AS data,
                array_agg(1 ORDER BY id) AS values
            FROM labs
                ASOF JOIN dynamic_data_feature_vocab ON 
                    labs.code = dynamic_data_feature_vocab.code AND dynamic_data_feature_vocab.type = 'labs' AND
                    labs.num_res >= dynamic_data_feature_vocab.lower_bound
            WHERE dynamic_data_feature_vocab.text_res <> ''
            GROUP BY study_id, month 
        ), lab_values AS (
            SELECT 
                study_id,
                month,
                array_agg(dynamic_data_feature_vocab.id ORDER BY id) AS data,
                array_agg(
                    (labs.num_res - dynamic_data_feature_vocab.avg_value)/dynamic_data_feature_vocab.stddev
                    ORDER BY id
                ) AS values
            FROM labs
                ASOF JOIN dynamic_data_feature_vocab ON 
                    labs.code = dynamic_data_feature_vocab.code AND dynamic_data_feature_vocab.type = 'labs' AND
                    labs.num_res >= dynamic_data_feature_vocab.lower_bound
            WHERE dynamic_data_feature_vocab.text_res = ''
            GROUP BY study_id, month 
        ), months AS (
            SELECT study_id, unnest(generate_series(1,{months}))::INTEGER AS month
            FROM subjects
        ), dynamic AS (
            SELECT months.study_id, array_agg(
                    flatten([
                        COALESCE(diag.data, []),
                        COALESCE(proc.data, []),
                        COALESCE(lab_binary.data, []),
                        COALESCE(lab_values.data, [])
                    ]) ORDER BY months.month
                ) AS data,
                array_agg(
                    flatten([
                        COALESCE(diag.values, []),
                        COALESCE(proc.values, []),
                        COALESCE(lab_binary.values, []),
                        COALESCE(lab_values.values, [])
                    ]) ORDER BY months.month
                ) AS values,
                array_agg(months.month ORDER BY months.month) AS months
            FROM months
                LEFT JOIN diag ON months.study_id = diag.study_id AND months.month = diag.month
                LEFT JOIN proc ON months.study_id = proc.study_id AND months.month = proc.month
                LEFT JOIN lab_binary ON months.study_id = lab_binary.study_id AND months.month = lab_binary.month
                LEFT JOIN lab_values ON months.study_id = lab_values.study_id AND months.month = lab_values.month
            GROUP BY months.study_id
        )
        SELECT
            nextval('data_idx_seq') AS idx,
            subjects.id, 
            static.data AS static_data,
            dynamic.data AS dynamic_data,
            dynamic.values AS dynamic_values,
            dynamic.months AS months
        FROM subjects
            INNER JOIN static ON subjects.study_id = static.study_id
            INNER JOIN dynamic ON subjects.study_id = dynamic.study_id
    """)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process data for the dataset for the diabetes A1C1 level prediction task')
    parser.add_argument('config', type=str, help='path to the configuration file')
    args = parser.parse_args()
    if args.config == ""  or not os.path.exists(args.config):
        print('Configuration file does not exist')
        sys.exit(1)

    conf = process_config(args.config)
    print('DataSet preprocess configuration:', conf)
    process_data(conf)