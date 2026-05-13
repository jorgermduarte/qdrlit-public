# Automatic Index Tuning via Quantum Deep Reinforcement Learning

SMARTIX: [https://doi.org/10.1007/978-3-031-42941-5_45](https://doi.org/10.1007/978-3-031-42941-5_45)

QRLIT: [https://www.mdpi.com/1999-5903/16/12/439](https://www.mdpi.com/1999-5903/16/12/439) 

DRLIT, QDRLIT: [https://www.mdpi.com/2504-4990/8/5/130](https://www.mdpi.com/2504-4990/8/5/130) 


[![License](https://img.shields.io/github/license/Chotom/rl-db-indexing)](https://github.com/Chotom/rl-db-indexing/blob/main/LICENSE)

---

## Setup and training

How to train an agent and set up an environment for TPC-H database.

1. In [db_env/tpch/tpch_tools](db_env/tpch/tpch_tools) directory paste tpc-h-tool.zip downloaded from TPC website (tested with version
   3.0.0): [download link](http://tpc.org/tpc_documents_current_versions/download_programs/tools-download-request5.asp?bm_type=TPC-H&bm_vers=3.0.0&mode=CURRENT-ONLY).

2. Define database size (in GB) and number of parallel streams in [db_env/tpch/config.py](db_env/tpch/config.py).

   You can check recommended number of streams in TPC-H documentation [here (page 96)](https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-H_v3.0.0.pdf#page=96). 
    ```python
    SCALE_FACTOR = 0.1   # Should be >=1, however we used 0.1 because of limited resources
    STREAM_COUNT = 2     # 2 for 1 GB, we also used 2
    ```

3. Follow further instruction:
   - Docker:
      - [Setup and training with Docker](#Setup-and-training-with-Docker)
   - without Docker:
     - [Server setup without Docker](#Server-setup-without-Docker)
     - [Client setup without Docker](#Client-setup-without-Docker)

---

### Setup and training with Docker

This should only be used for testing or if you are certain, that you can provide stable server performance.

1. Start mysql server and client.
    ```shell
    docker compose up -d
    docker compose up -d --build # if docker images require to be rebuilt or created
    ```

2. Generate data and load database to mysql_server from client container.
    ```shell
    docker compose exec client python3 /project/cli/initiate_environment.py
    ```

3. You can start training by running script in client container
    ```shell
    docker compose exec client python3 /project/cli/run_train.py
    ```

---

### Server setup without Docker

1. Install and configure mysql server (version 8).

2. Allow remote root connection for database operations during benchmarking (or create different user):
    ```MySQL
    CREATE USER 'root'@'CLIENT_IP'  /* change CLIENT_IP to client IP address, or use 'root'@'%' for all addresses */
    IDENTIFIED WITH caching_sha2_password BY '1234';
    GRANT ALL PRIVILEGES ON *.* TO 'root'@'CLIENT_IP';  /* change CLIENT_IP */
    ```
---

### Client setup without Docker

Project currently only supports Linux client, because of DBGen. However, it should be possible to compile
DBGen on Windows as well.

1. Update file [.env](.env) with MySQL server data (user, password, server IP, port and name of the database, which will be
used for benchmarking).

2. Install gcc, make, python (minimal version is 3.8), pip and virtualenv.

3. Move project to desired folder (for example /path/to/project).

4. Create virtual environment (for example inside /path/to/project directory):
    ```shell
    virtualenv venv
    ```

5. Save project path to $PYTHONPATH:
    ```shell
    echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/project/"' >> venv/bin/activate
    ```

6. Activate virtual environment:
    ```shell
    source venv/bin/activate
    ```

7. Install requirements:
    ```shell
    pip install -r requirements.txt
    ```

8. Patch DBGen:
    ```shell
    python cli/patch_dbgen.py
    ```

9. Compile DBGen:
    ```shell
    make -C dbgen
    ```

10. Initiate environment - populate database with generated data:
    ```shell
    python cli/initiate_environment.py
    ```

The environment is now ready and you can:

- train agent, for example:
    ```shell
    nohup python cli/train_agent.py &> data/train.log &
    ```

    It is possible to stop training at any time (without need of resetting environment) by sending the script SINGINT
    (CTRL+C). In case of example way of running training shown above, it can be done with:
    ```shell
    kill -2 PID   # change PID to process id (can be established using 'ps' command)
    ```

- run single benchmark on current database configuration:
    ```shell
    python cli/run_benchmark.py
    ```

- reset database (without generating data for benchmark again):
    ```shell
    python cli/reset_environment.py
    ```

- reset index configuration to default state:
    ```shell
    python cli/reset_indexes.py
    ```

- set specific index configuration (run `python cli/set_index.py --help` to see column order):
    ```shell
    python cli/set_index.py 100000000000000000000000000000000000000000000
    ```
    
---

## Important notes

### DBGen bug

In file BUGS in dbgen directory in official TPC-H Tools, a bug named "Problem #00062" is mentioned, which states the
following:

```
bad update rollover after 1000 refreshes
This test uses tpcH scale 0.01. We've encountered
an situation in which dbgen doesn't generate
the correct data for delete files delete.1000 and
above. In particular, file delete.1000 contains
keys to be deleted that have never been loaded.
Because of this problem, keys that should have been
deleted never are causing duplicate unique values
to appear in the incremental loads after we cycle
from the 4000th incremental update back around starting
again with the 1st one.
```

The bug was closed, supposedly due to an unsupported scale factor. However, according to our observations, the bug still
exists and affects every SCALE_FACTOR.

Due to this bug database is constantly growing in size after 999 refresh functions. In order to continue with
experiments we had to implement a "fixed" way of generating data.

Our way of "fixing" data uses only 1998 refreshes, after which the database is in its initial state (as opposed to 4000, which
would be official number if the bug hadn't existed).

Had there been any solution for that bug, ensure to update MAX_REFRESH_FILE_INDEX variable in the file
[db_env/tpch/config.py](db_env/tpch/config.py).

```python
MAX_REFRESH_FILE_INDEX = 4000 # 1998 changed to 4000
```

and file [cli/initiate_environment.py](cli/initiate_environment.py) like so:
```python
from db_env.tpch.TpchGenerator import TpchGenerator

if __name__ == '__main__':
    generator = TpchGenerator()
    generator.reset_db()
    generator.generate_data()
    generator.load_db()
    generator.generate_refresh_data()           # uncomment this line
    #generator.generate_fixed_refresh_data()    # comment or remove this line
```

#### How to confirm the bug

1. Generate data with DBGen (note officially supported 1 GB SCALE_FACTOR):
    ```shell
    ./dbgen -vf -s 1
    ```

2. Generate refresh data for 4000 refresh function pairs:
    ```shell
    ./dbgen -vf -U 4000 -s 1
    ```

3. Delete files from 1000 onwards do not delete anything, which can be checked by:
    ```shell
    grep -F -f delete.1000 ./orders.tbl* 
    ```

This demonstrates that, after refreshes 1-999 (which are assumed to be correct), delete files don't actually remove
anything. At the same time update files do add records, which causes database to grow in size, which messes up the
benchmark. 

### Database state file

You should never interfere with file path/to/project/data/rf_db_index.txt.

Every executed refresh pair should increment number stored in that file.

After refresh pair number 4000 (or currently after refresh pair 1998) database should be in its initial
state (note this has not been fully tested) - rf_db_index.txt should contain 1.

Otherwise, you will lose information about current database state, which may (and sooner or later will) lead to errors
and force you to stop training and reset database.

---

## Project structure

### agent

Package with agent to train.

### data

Directory to store project datafiles and agent data for analysis.

### data_analysis

Directory to store scripts used for analysis of agent training data.

### db_env

Package with database environment for agent.

### shared_utils

Package with utilities functions and variables to use in project.

### test

Run tests with code coverage (run `pip install -r test/requirements.txt` to install test requirements):

```shell
coverage run -m unittest
coverage report
```

---