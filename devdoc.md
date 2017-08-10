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
1. The master will build both the local and global graph. Note that for replication purposes, the master is the node which initializes and replicates all global graph data/ops to the workers. For local graphs, each node manages its own isolated data/ops. The master calls `manylmc.optimizeGlobal()` which is a wrapper around the actual logic, `lmc._sgd_Master()` which executes the global training loop.


## Worker
TODO

## Synch & Async Policies
###### ref: `lmc_joint_model.py` Search: @synchpolicy  
  
The policies are defined via the program flag `FLAGS.synch_method`. They are processed by the master in `_sgd_master` and by the workers in `_sgd`. For the workers: the policy is mostly for logging purposes to ensure the log output reflects the correct timestep and to set the max # of iterations (for synch training). The master does pretty much all the the work.

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
* The gradient TF queue is a tuple for worker, timestamp, the gradient. Each iteration of the loop, the master locally stores these in a the list `incoming` (line ~3385)
* For each grdient received, the master either accepts or rejects it (unless the worker flags that it is complete).
* Master then takes the action based on the synch method (lines 3435~3460). This is where the master either stores it for processing in a TF aggregator variable (using the method `storeGrad`) or holds it in a python sortedList (`jacPQueue`) until its applied. Note that there is index (`jacCache`) into this list to facilitate processing. The sortedList is part of the python sortedcontainers package. The sorting function for each policy is defined as a lambda function (lines 3290-3300)
* If a gradient decision is needed (line 3479), the actual parameter update is applied via the `applyGrad` call on line 3524. For the priority queue based policies (based on staleness or cosine distance), the master first prunes gradients, pops the top-A gradients, and then stores them in the TF aggregator variable.
--> NOTE: The sorting policies are managed on the python side since maintaining a sorted list is very difficult using TF ops.

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


## Global Model

## Local Model

## Data Interface

## 

