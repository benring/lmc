# Developer's Document

### Execution
* Python-based application (v2.7.12) using TensorFlow (v1.0.1)
* Additional dependencies are found in `requirments.txt`
* The CustomOp is written in C++: There is a separate developer's document for this
* Main entrypoint for training:  `joint_model.py`
* Main entrypoing for testing: `online_prediction.py`

### Design Overview
TODO (if needed) 

## Environment
###### ref: `environ.py`, `tools/timelog.py`
1. **`Environment` Class**. This class provides a means to share data and state among all classes/methods contained within a single executing process. Since it is a `@Singleton`, the application will only instantiate one object which is the first line of code in `__main__`. Any subsequent call for instantiation will always return the same object. 

2. **Program Options**. The `setFlags()` method is where you define program options. It uses the TensorFlow flags module which defines key-value fields for boolean, integer, float, and string and will set the corresponding option if the command line option is provided. For the default values, I use the pattern, `os.getenv("FLAG_NAME") or {value}`. This will initially try to set the option using an environment variable and if not set, then apply the given default. 
--> *Key take-away:* command line options have a higher precedent over environment variables for setting program options.
One addtional note: you can always overwrite these values dynamically

3. **Configuring**. Use the `config()` method to initialize the program options. the `tf.app.flags` will set all program options when the first call to this module is invoked. By consolidating everything in `setFlags()` this provides control over when program options are set. This also performs an autocheck to detect if the program is running in AWS. The best way is to connect to Amazone instance URL using the following:
```
        # Detect if operating within AWS EC2 instance
        try:
            _ = urllib2.urlopen('http://instance-data.ec2.internal')
            self.FLAGS.aws = True
        except urllib2.URLError:
            self.FLAGS.aws = False
```

4. **Logging**. The `config()` method also sets up the logging for the program. It is set up to have workers log to a local file called `lmc.log` and the master to `lmc.ps`. The actual logging is set up in the file, `tools/timelog.py` which configures both the file and console logging using a time relative to the start of program execution (specifically, the time starts when the elapsed formatter is instatiated, but the difference is a few milliseconds). To change the log format, modify the method, `format()` in the `ElapsedFormatter` class inside this file. Default logging level is set up for debug, but  you can change this here.
--> *Note*: Do not confuse the logging level with the verbose program option -- they are separate. The verbose option is used to print additional debugging data and is covered below (it does not affect the logging level). 

## Execution (General)
###### ref: `joint_model.py`
1. **Main**. The `__main__` methods configures the environment and calls the `tf.app.run()` which, in turn calls `main(_)` -- if needed, you can set up tf.app.run to invoke a different method and pass-thru different (or modified) command line options. 

2. **Exception Handling**. There is a catch-all exception included in the main method. In addition, exception handling is including at key points in the code. In some cases, the exceptions are commented out but can be very useful for debugging. If running in k8s, it will enter an infinite loop and print `Please kill me` -- see termination below for more info on this statement. 

3. **Single Server Mode**. The codebase was originally designed to execute in either single server mode or in a distributed mode. (Note that single-server mode is different than running a master and a worker process on the same host. The former only executes one process on one port, the latter executes two processes using different ports). However, due to the added burden of maintaining both branches, I did not keep up the single-server implementation. I did not remove the logic for this mode (located in various points in the codebase) in case someone wanted to go back and update this later on. It will take some design work to go back and combine the worker and the master components back into one.
 
4. **Hostname**. (search: *@hostnameparse*) The program is configured by default to automatically configure the TensorFlow distributed cluster hostnames simply by setting the number of workers. If you are not running in k8s, then you should set num_workers to 0 and provide the actual list of master/workers with port, just like the TF tutorials. In k8s, the program assumes there is only one master and this master will run on pod-0. All workers will be on pod-1 thru pod-N. This provided deterministic assignement of roles within the cluster, facilitating the job/index assignement. 
--> The fully qualifuied hostname is statically defined in the codebase as `{statefulset_name}-{index}.tensorflow-svc:2222`. This coincides with the k8s service defined in the input yaml file. If you want to change this or the port, you can make this a dynamically assigned name or read it in from a static value or program option. Either way it must match the k8s provided host and service name.
--> Workers are internally numbered with a ZERO-based index, but the hostnames for the workers start at pod-1 (since the master is on pod-0). Example: worker #0 will run on hostname `tflmc-1`

5. **Cluster Configuration**. (Search: @clusterconfig) There is a uniCluster object to manage the device placement. This was originally designed to consolidate cluster management and make it easier to set up GPU/CPU and param server logic. However, it's probably not necessary any more. Otherwise, the global session is configured with a TF supervisor which can accept TF options for checkpointing and outputting summary statistics. I did focus much on getting the checkpointing working and I'm not sure if it would even work with the GPFlow components, but if you want to try and get checkpointing to work with fault tolerance, then update/modify the `globalSaver` object. For fault tolerance, you may also want to look at the localSaver object.

6. **Postpartum Processing**. There are seprate master and worker methods for all logic which should occur after the training/testing is complete. This mainly includes storing metrics in the database and copying all local files (e.g. logs) to a central location.

## Master
###### ref: `lmc_joint_model.py`
1. 


## Worker
TODO

## Synch & Async Policies
###### ref: `lmc_joint_model.py` Search: @synchpolicy  

The policies are defined via the program flag `FLAGS.synch_method`. They are processed by the master in `_sgd_master` and by the workers in `_sgd`. For the workers: the policy is mostly for logging purposes to ensure the log output reflects the correct timestep and to set the max # of iterations (for synch training). The master does pretty much all the the work.

--> Note: The master will build both a local and the global graph. For replication purposes, the master is the node which initializes and replicates all global graph data/ops to the workers. For local graphs, each node manages its own isolated data/ops. The master calls `manylmc.optimizeGlobal()` which is a wrapper around the actual logic, `lmc._sgd_Master()` which executes the global training loop. This method is where the bulk of the sync/async policies are implemented.


#### Asynchrony 
* The actual asynchrony value to set is configurable via the program flag `synch_block_size`. This determines the # of gradients aggregated.  In addition, there is a flag for `asynch_max` which is the number of gradients accepted before aggregation (note: `asynch_max  >= synch_block_size `). This flag acts as a delay factor for the master to wait before applying gradients. 
* The codebase is designed such that this can be dyamically changed. There are 3 variables that affect asychrony which can change:
```
     max_toAccept - max # grad held in local (python) based list for processing at any step
     min_toAccept - min # grad required before making decision on what to process
     num_toApply  - # grads to actually apply
     Typically:  num_A <= min_A <= max_A
```
For FIFO and synch policies `num_A = min_A` - setting the min_A any higher for these policies would be moot. 

#### Gradient Processing
* At the top of the main master loop (line ~3375), The master dequeu's 1 or more gradients to start its processing loop. We set a value for `max_accept` based on the gradient policy (lines 3140-3160), but, in general, the program will only process up to num_worker gradients at a time. This ensures it does not over-aggregate in the event of network latency when the queue may build up and it allows the master to relieve pressure by processing more gradients than the asynchrony value when the queue is very large.
* The gradient TF queue is a tuple for (worker, timestamp, gradient). Each iteration of the loop, the master locally stores these in a the list `incoming` (line ~3385)
* For each grdient received, the master either accepts or rejects it (unless the worker flags that it is complete).
* Master then takes the action based on the synch method (lines 3435~3460). This is where the master either stores it for processing in a TF aggregator variable (using the method `storeGrad`) or holds it in a python sortedList (`jacPQueue`) until its applied. Note that there is index (`jacCache`) into this list to facilitate processing. The sortedList is part of the python sortedcontainers package. The sorting function for each policy is defined as a lambda function (lines 3290-3300)
* If a gradient decision is needed (line 3479), the actual parameter update is applied via the `applyGrad` call on line 3524. For the priority queue based policies (based on staleness or cosine distance), the master first prunes gradients, pops the top-A gradients, and then stores them in the TF aggregator variable.
--> NOTE: The async sorting policies are managed in Python since maintaining a sorted list using TF ops is very difficult.

#### Convergence & Termination
* Termination is currently handled by stopping at a pre-determined number of iterations. For synch policy, each worker maintains a running count and stops when it reaches the `max_svi_iter` value. For the asynch policy, the master determines the total number of gradients and then flags the workers to stop training. This value is `num_workers X max_svi_iters` and is set up to be the same total gradients applied as compared to a synch policy. Note that the master will still process gradients until the workers report back that they have actually received the terminatin flag and have stopped their local logic (the flag is simply setting the timestamp to -1).
* *If you want to terminate based on convergence*: There is already commented out place-holder logic to calculate convergence (we just never fully tested and developed it). See lines 3552-3561. 

#### Policy List
* **synch**: Fully consistent, synchronized (everything else is async)
* **fifo**: FIFO
* **asynch**, **asypq**: (both the same thing). These use the Least Stale First ordering for the gradients
* **stale**: Most Stale First
* **cdismin**, **cdismax**: Uses Cosine Similiarity prioritizing by either the min or max distance.
* **granular**: Applies gradients one at a time in FIFO order -- this was designed to perform a different set of TF Ops which are more optimized for this policy. I haven't tested this much, so it may not work properly. You can set the asynchrony level to 1 and get the same effect (but not as optimized as this was designed to be).
* **wskew**: Experimental policy that weighted gradients based on the data skew of the workers.


## Data Interface
###### ref: `tools/db.py`, `datainterface.py`

There are two files to handle the data processing and I/O. The file, `db.py` (in the tools directory), was the original file and it was designed as a very light-weight self-contained wrapper around psycop2g. It's written so that you can import it as a standalone file into an interactive or jupyter session to help with analysis, processing or development.  The file, `datainterface.py` was written much later and is designed to serve as an abtraction layer in between the LMC program and the lower level database I/O. It's not a 100% clean separation but it should be much easier to integrate a new interface for a different dataset (it currently has interfaces for Mimic and CDM for both online and offline access).

### `db.py`
* Provides a wrapper class `db_conn` to handle database connections.  The connection strings to use SSL are included (but commented out). There are also predefined connection strings for the test/dev/prod databases on AWS.
* This module maintains an active state for the current default connection via `db.DEF_CONN`. All functions/methods will use this connection string unless otherwise specfied.
* Use `setDefaultDB()` to change this value
* The methods, `adhoc()` and `runquery()` provide all the data I/O to run a single query. The second returns a result and the first does not. Note: these are designed for smaller queries (small as in input/output size).

##### DDL & Tables
The DDL is also defined in this file. I used a prefix in case there was ever a need to have two sets of the table or we wanted to coordinate the table names as part of a larger effort. For the most of these tables, there are pre-defined get/set methods (as needed) for data in/output. However, some of the I/O ops are found in the `datainterface.py` file.

* **model** : Data for a trained model
* **expr** : Data for an experiment or a single "training session." The model and experiment (and associated mid/eid) were designed to be separate in case we wanted to re-run training using the same model. However, this is no longer necessary and they can be combined.
* **param_global** : The global parameters, retained at each step and stored as a BLOB.
* **param_local** : For storing the local params for each patient's data. We never used this for training, but this is how you can checkpoint patients' local parameters for fault tolerance.
* **metric** : Key-value table to store various metrics collected during training (these metrics mostly helped with the research experiments)
* **score** : stores prediction output
* **predict** : also store prediction output (but for Peter)
* **kv** : A generic key-value table for storing json formatted data. All program flags for every experiement is stored in this table along with the shard lists for every dataset.
* **numpy** : Generic key-value for storing numpy formatted data. Its primarily used to store intermediate node scores after prediction. The master pulls/aggregates all data and outputs results to the score table.
* **mintsp** : stores min timestamp for all encounter ID's (by dataset). This is used to convert timestamp to time epochs used in LMC.


### `datainterface.py`
* Three high level abstractions:
  1. Patient object
  2. Feature definition
  3. Data interface class <-- this is where most of the meat is found

 ##### Feature Definition
`FeatureDef` encapsulates and abstracts the feature logic into one convenient location. Currently, pre-determined features are hard-coded into the class (`feature_options` list).  The program flag, `feat_size` is literally an index into this list (items 0-3 are mimic and items 4-6 are CDM). The subclasses perform dataset specific pre-processing for the features to define the static/dense subsets and to put the features in a very specific order for the LMC model. This is also a placeholder for future expansion to create custom feature defs for different datasets or different tasks.

 ##### Data Interface
 * Abstract class for encapsulating all data logic between LMC and the database. It is designed to support both SQL and File-based I/O. 
 * The object `self.data` holds the raw data loaded for the local node. 
 * `self.data` was originally a pandas dataframe for mimic data. I later changed this to a numpy array. For the online prediction, I expanded the capability to make it a map {enc_id --> raw_data}. This allows the local node to more delete and add patients dynamically and is what is used in online_predicion.
 * If you want to make this more memory efficient during training, you can delete the raw data object after the data is transformed to tensors.
 * For CDM subclass: the `loadPatientData` is the call to actually invoke the dashan-db implemented connection to read in all the patient data. This is the one-and-only dependency on the dashad-db codebase. If you want to break this dependency, you can re-write this portion of the code.
 * **Labels**. `loadLabelData` loads all the patient labels using match_patient_encounters stored procedure. The function, `applyLabelToData` using a simple lambda step function to assign a boolean label to every item in a patient's dataset. The LMC model is designed such that a patient is considered "positive" for all timestamp readings after the first diagnosis of sepsis (or septic shock).
 * Active "state" for a patient is defined in the following objects:
   * `self.data[encid]`  &rarr; Raw Data (the mapping is used for online prediction)
   * `self.mintsp[encid]` &rarr; Min Timestamp. MIN of [admit_time, admission time, or first data reading] 
   * `self.enc_tsp[encid]` &rarr; Last encounter timestamp -- used for delta query in online predicion
   * `self.pt_map[encid]` &rarr; map to the patient object which holds the transformed data used to create the tensor objects as part of training. 
   * `self.pt_tensor_map[encid]` &rarr; Added later and used in online prediction. This is a mapping to the tensor objects. This is a much more efficient way to manage the tensors to pass into the TF graph. If you want to refactor the training side to use this, you can.
* The `removePatients` method is designed to accept a current list of active patients. It compares that list with the raw patient data list in cache and delete the difference.
