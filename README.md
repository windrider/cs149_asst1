# CS149

#### 简介	

​	**stanford CS149**并行计算课程的_assignment1_，一共有6个_program lab_。

​	课程网址：https://gfxcourses.stanford.edu/cs149/fall22

​	原**assignment**网址：https://github.com/stanford-cs149/asst1

## Program 1: Parallel Fractal Generation Using Threads

​		要求使用多线程并行计算将_mandelbrot_集合中的点转化为_ppm_格式的图像。_main_函数中使用顺序计算和并行计算两种方式来计算出_mandelbrot_图像，比较两种方式的速度，其中串行计算的代码已经给出，要求实验者自行实现并行计算。源代码中计算_mandelbrot_图像上所有点的像素值的函数为：

```c++
void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int totalRows,
    int maxIterations,
    int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    int endRow = startRow + totalRows;

    for (int j = startRow; j < endRow; j++) {
        for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = mandel(x, y, maxIterations);
        }
    }
}
```

​		显然该函数是从_startRow_到_endRow_按行从上到下计算出每一行的像素点，一共有_totalRows_行。要优化这个算法，可以用多个线程执行该函数，每个线程分别计算图像的一部分，并行计算出完整的图像。

```c++
int step=height/numThreads;
for (int i=0; i<numThreads; i++) {
      
        // TODO FOR CS149 STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;
      
        args[i].threadId = i;
        args[i].startRow=i*step;
        args[i].numRows= (i==numThreads-1?height-i*step:step);
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i=1; i<numThreads; i++) {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }
    
    workerThreadStart(&args[0]);

    // join worker threads
    for (int i=1; i<numThreads; i++) {
        workers[i].join();
    }
```

​	多线程并行计算的实行方法如上，初始化_numThreads_个线程，图像的总高度为_height_,令步长_step=height/numThreads_,每个工作线程从不同的 _startRow_ 开始计算_numRows_行，除最后一个线程外，每个线程要计算的_numRows=step_，但最后一个线程必须要计算完所有剩下的行数。

​	设计线程的参数结构体和线程函数_workThreadStart_,并加上计时的代码，编译运行后统计不同线程数耗费的时间，列表如下：

| 线程数 | 运行时间(ms) |
| ------ | ------------ |
| 1      | 328.657      |
| 2      | 170.309      |
| 3      | 205.676      |
| 4      | 140.183      |
| 5      | 138.016      |
| 6      | 111.314      |
| 7      | 111.284      |
| 8      | 107.583      |

​	发现总体趋势是线程数越多，程序的运行时间越少。但是线程数大于4以后，增加线程的优化效果就很弱了，这是因为我们的cpu只有4核，线程数超过4后就会有线程需要等待cpu资源。另外发现线程数从2到3，程序的运行时间不降反增。为了探求这个问题，打印每个线程的运行时间，由于计算运行时间的逻辑是取5次运行中耗时最小的那次，所以打印的线程情况会循环5次。

![ce15db0f840e26ef585ced3f8cf86c7](.\graphs\ce15db0f840e26ef585ced3f8cf86c7.png)

![269f0d7b942fa4099cb0d4f10e8df1e](.\graphs\269f0d7b942fa4099cb0d4f10e8df1e.png)

​	可以发现2个线程时，每个线程的运行时间是均匀的，但是3个线程时，1号线程运行时间远高于0号线程和2号线程。这是因为虽然每个线程都平均分配了_step_行，但是计算量却不一样。由于_mandelbrot_图像的特点是每个像素的亮度都和计算该像素的复杂度正相关，图像的中间远比上下两侧更亮，1号线程刚好被分配到了计算该图像的中间区域，所以计算耗时远远大于另外两个线程，也就拖慢了整体的线程运行时间。所以使用多线程并行计算时，应该尽量使每个线程的计算量均匀。



## Program 2: Vectorizing Code Using SIMD Intrinsics 

​	Program2要求用模拟的SIMD指令优化两个函数，一个是计算乘方的**clampedExpSerial**函数，另一个是计算数组和的**arraySumSerial**函数。

​	模拟的SIMD指令在**cs149intrin.h**中给出了定义。SIMD指令就是利用架构中的矢量寄存器，让一条指令能同时计算寄存器中的多个元素，达到优化目的。SIMD指令的形式和语义和汇编语言相似。	**cs149intrin.h**中用**vector<bool>**模拟了**__cs149_mask**类型，实现掩码的语义。该实验中大部分指令都有**_cs149_mask**类型的参数。只有当**__cs149_mask**中对应位为1时，SIMD指令才能在矢量寄存器的对应位置上生效。实验中发现**__cs149_mask**类型在赋值，计算和条件判断上都能发挥作用。

​	优化后的函数如下：

```C++
void clampedExpVector(float* values, int* exponents, float* output, int N) {

  //
  // CS149 STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  
  __cs149_vec_float x;
  __cs149_vec_int   y;
  __cs149_vec_float result;
  __cs149_vec_int count;
  __cs149_vec_float floatZero = _cs149_vset_float(0.f);
  __cs149_vec_float floatOne  = _cs149_vset_float(1.f);
  __cs149_vec_float float9p9  = _cs149_vset_float(9.999999f);
  __cs149_vec_int intOne  = _cs149_vset_int(1);
  __cs149_vec_int   intZero = _cs149_vset_int(0);
  __cs149_mask maskAll, maskIsNegative,maskIsZero, maskNotZero,maskIsNotNegative,maskCountIsZero,maskCountNotZero,
  maskResultGt9p9;
  
  for(int i=0;i<N;i+=VECTOR_WIDTH){
      maskAll = _cs149_init_ones();
      maskIsNegative = _cs149_init_ones(0);
      maskIsZero = _cs149_init_ones(0);
      _cs149_vload_float(x, values+i, maskAll);  
      _cs149_vload_int(y,exponents+i,maskAll);
      _cs149_veq_int(maskIsZero, y, intZero, maskAll); //if y==0
      _cs149_vset_float(result,1.f,maskIsZero);   //output=1
      maskNotZero=_cs149_mask_not(maskIsZero);  //else
      _cs149_vload_float(result,values+i,maskNotZero); //result=x
      _cs149_vsub_int(count,y,intOne,maskNotZero);  //count=y-1;
      _cs149_veq_int(maskCountIsZero, count, intZero, maskAll); 
      maskCountNotZero=_cs149_mask_not(maskCountIsZero);
      
      while(_cs149_cntbits(maskCountIsZero)!=VECTOR_WIDTH){  //while (count>0)
          _cs149_vmult_float(result,result,x,maskCountNotZero); //result*=x
          _cs149_vsub_int(count,count,intOne,maskCountNotZero); //count--
          _cs149_veq_int(maskCountIsZero, count, intZero, maskAll);
          maskCountNotZero=_cs149_mask_not(maskCountIsZero);
      }
      
      maskResultGt9p9=_cs149_init_ones(0);
      _cs149_vgt_float(maskResultGt9p9,result,float9p9,maskNotZero); //if result>9.999999f
      _cs149_vset_float(result,9.99999f,maskResultGt9p9); //result=9.999999f
      _cs149_vstore_float(output+i,result,maskAll); //output[i]=result
  }
}

```

```c++
float arraySumVector(float* values, int N) {

  //
  // CS149 STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  
  float sum[VECTOR_WIDTH];
  __cs149_vec_float result=_cs149_vset_float(0.f);
  __cs149_vec_float x;
  __cs149_mask maskAll=_cs149_init_ones();;
  for (int i=0; i<N; i+=VECTOR_WIDTH) {
    _cs149_vload_float(x,values+i,maskAll);
    _cs149_vadd_float(result,result,x,maskAll);
  }
  _cs149_hadd_float(result,result);
  _cs149_interleave_float(result,result);
  _cs149_hadd_float(result,result);
  _cs149_vstore_float(sum,result,maskAll);

  return sum[0];
  
 //return 0.0;
}
```

​		编译后计算10000个元素的数组测试通过。

![0b1b5f838fd7de121fd7ee77abd63c2](.\graphs\0b1b5f838fd7de121fd7ee77abd63c2.png)



## Program 3: Parallel Fractal Generation Using ISPC

### Program 3, Part 1. A Few ISPC Basics 

​	使用ISPC加速计算_mandelbrot_图像，并和顺序计算_mandelbrot_图像的时间做比较。ISPC是位高性能SIMD编程设计的编译器，ISPC程序的多个程序实例是总是在CPU的SIMD执行单元上并行执行。ISPC文件会编译成.o文件和.h文件，c文件在**#include**对应的.h文件后，可以直接调用ISPC文件中编写的函数。

```C++
export void mandelbrot_ispc(uniform float x0, uniform float y0, 
                            uniform float x1, uniform float y1,
                            uniform int width, uniform int height, 
                            uniform int maxIterations,
                            uniform int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    foreach (j = 0 ... height, i = 0 ... width) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = j * width + i;
            output[index] = mandel(x, y, maxIterations);
    }
}
```

​	![1475161285be8839dd9d16a7be6ab49](.\graphs\1475161285be8839dd9d16a7be6ab49.png)

​	编译运行,发现使用ISPC使计算速度提升了5倍。

### Program 3, Part 2: ISPC Tasks 

​		使用**ISPC_task**机制,通过**launch**指令启动多个**task**，在多个核上并行运行。本实验采用和Program1一样的思路，按行从上到下平均划分**task**。

```C++
task void mandelbrot_ispc_task(uniform float x0, uniform float y0, 
                               uniform float x1, uniform float y1,
                               uniform int width, uniform int height,
                               uniform int rowsPerTask,
                               uniform int maxIterations,
                               uniform int output[],
                               uniform int taskNum)
{

    // taskIndex is an ISPC built-in

    uniform int ystart = taskIndex * rowsPerTask;
    uniform int yend = (taskIndex== taskNum-1? height : ystart + rowsPerTask);
    //uniform int yend = ystart + rowsPerTask;
    
    uniform float dx = (x1 - x0) / width;
    uniform float dy = (y1 - y0) / height;
    
    foreach (j = ystart ... yend, i = 0 ... width) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;
            
            int index = j * width + i;
            output[index] = mandel(x, y, maxIterations);
    }
}

export void mandelbrot_ispc_withtasks(uniform float x0, uniform float y0,
                                      uniform float x1, uniform float y1,
                                      uniform int width, uniform int height,
                                      uniform int maxIterations,
                                      uniform int output[],
                                      uniform int taskNum)
{

    uniform int rowsPerTask = height / taskNum;

    launch[taskNum] mandelbrot_ispc_task(x0, y0, x1, y1,
                                     width, height,
                                     rowsPerTask,
                                     maxIterations,
                                     output,taskNum); 
}
```

![2716f4640d52d4bed66deb3f29714d6](.\graphs\2716f4640d52d4bed66deb3f29714d6.png)

![dd7895cf13f68f25309cd6f0313c49f](.\graphs\dd7895cf13f68f25309cd6f0313c49f.png)

发现4个task能使速度提升为原来的13倍，8个task能使速度提升为20倍，速度提升的倍速小于理想的等比例提升，可能是由于图片上像素点的计算复杂度不是均匀的，并且切换task会产生开销。



## Program 4: Iterative `sqrt` 

​		优化平方根算法。程序会计算两千万个0~3之间的随机数（的倒数？）的平方根，使用的方法是牛顿迭代法，求解${\frac{1}{x^2}} - S = 0$这个方程的零点，并将初始的**guess**值设为1

​		比较使用**ISPC**和使用**ISPC TASK**对程序的速度提升，发现使用**ispc**将速度提升了5倍，使用**ISPC TASK**（64个task）将速度提升了18倍。

![56e48af02887830d6abfe02f766bb1a](.\graphs\56e48af02887830d6abfe02f766bb1a.png)

​		

```c++
void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}

```

​		这代码很显然是算倒数的平方根	

#### 修改输入使原顺序计算版本的算法速度最快		

​	很显然将两千万个输入都设为1，那么初始的**guess**值恰好就是1，不用迭代，此时的计算速度为

![c79e8c4708e3c5a67c81951b464cedb](.\graphs\c79e8c4708e3c5a67c81951b464cedb.png)

​	此时使用**ISPC**和**ISPC TASK**都只能将速度提升2倍，并且是否使用**TASK**对结果影响不大

#### 修改输入使原顺序计算版本的算法速度最慢

​	很显然将两千万个输入都设为0，理论上程序会直接爆炸，但可能因为浮点数并不是完全精确的，所以最后居然能算出结果。比较使用**ISPC**和使用**ISPC TASK**对程序的速度提升，发现使用**ispc**将速度提升了7倍，使用**ISPC TASK**（64个task）将速度提升了28倍。

![d76456e7e1d6a31e9002dca237951af](.\graphs\d76456e7e1d6a31e9002dca237951af.png)





另外，如果将两千万个输入都设为0.333333f,运行结果是这样的：

![e15215fb54840b078bb57d29dd424e8](.\graphs\e15215fb54840b078bb57d29dd424e8.png)

速度依然比随机输入两千万个0~3之间的数要快



所以可以发现，修改算法不同的输入，当使原顺序算法的计算速度最快时，使用**ISPC**和**ISPC TASK**对程序的优化效果越不明显。另外可以推出SIMD和多任务多核进行并行计算，都希望各个任务的平均计算量基本一致，当并行的多个任务计算时间不一致时，最终的计算时间会被计算时间最久的那个任务严重拖慢（所以两千万个随机数的计算时间会很慢）。



## **Program 5: BLAS `saxpy`**

​		加速saxpy,即 计算`result = scale*X+Y`，编译运行使用了**ISPC**和**ISPC_TASK**机制的程序

![b16747f433a84797bbb5eb391df021f](.\graphs\b16747f433a84797bbb5eb391df021f.png)

```C++
task void saxpy_ispc_task(uniform int N,
                               uniform int span,
                               uniform float scale,
                               uniform float X[], 
                               uniform float Y[],
                               uniform float result[])
{

    uniform int indexStart = taskIndex * span;
    uniform int indexEnd = min(N, indexStart + span);

    foreach (i = indexStart ... indexEnd) {
        result[i] = scale * X[i] + Y[i];
     
    }
}
```

​	

​	发现使用了**ISPC_TASK**和仅仅使用**IPSC**相比提升很小，没有达到预期的性能。影响性能的原因有很多。例如Program5和Program4相比，计算时用到result,X,Y三个数组，虽然计算量更小，但是更频繁地读写内存严重影响性能。**main.cpp**中计算带宽的语句也暗示了本程序的性能瓶颈在于内存带宽的限制，不是开多线程就能解决的。

​	使用**ISPC_TASK**计算saxpy的程序中一共launch了64个task, 如果真的有64个线程，频繁切换线程上下文也会造成开销，并且也会弄脏缓存，造成更多的cache miss，也会影响性能，改为4个线程以后性能反而有略微的提升。而且task是比thread更高层次的抽象，ISPC编译器不一定真的安排了64个线程并行计算。

​	 关于**TOTAL_BYTES = 4 * N * sizeof(float)**，因为写result数组的时候需要对应的内存块更新到cache中，这里有1次读内存，1次写内存，读X,Y数组的时候又有2次读取内存，所以总共和内存交互的字节数是**4 * N * sizeof(float)**;

## Program 6: Making `K-Means` Faster 

​		找到K-Means算法的性能瓶颈，进行优化。

​		由于实验只允许修改KmeansThread.cpp文件，对该文件下的每个函数都计时输出：

![7db313149006f73fdd8b69c5867ad73](.\graphs\7db313149006f73fdd8b69c5867ad73.png )

![46d6290a9c9821558f1d69478f6ec10](.\graphs\46d6290a9c9821558f1d69478f6ec10.png)

   		 

​    发现总耗时要10s,并且负责将点分配给聚类的**computeAssignments**函数是耗时最长的。

​	 观察**comuteAssignments**函数：

```c++
/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args) {
  double *minDist = new double[args->M];
  int k=args->threadId;
  // Initialize arrays
  for (int m =0; m < args->M; m++) {
    minDist[m] = 1e30;
    args->clusterAssignments[m] = -1;
  }

  // Assign datapoints to closest centroids
  for (int k = args->start; k < args->end; k++) {
    for (int m = 0; m < args->M; m++) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      if (d < minDist[m]) {
        minDist[m] = d;
        args->clusterAssignments[m] = k;
      }
    }
  }

  free(minDist);
}
```
​	发现该函数主要执行了两层循环，k遍历所有的聚类，m遍历所有的点，都每个点找到距离最近的聚类中心。由于聚类的数量并不多，可以去掉最外层循环，改用多线程的方式，每个线程负责计算所有点到某个聚类中心的距离。

​	写下多线程计算所有点到聚类中心距离的算法,由于K个线程都要访问minDist，访问时加锁：	

```c++
void computeAssignments(WorkerArgs *const args,double* minDist, std::mutex& mtx) {
  // Initialize arrays
  int k=args->threadId;

  // Assign datapoints to closest centroids
    for (int m = 0; m < args->M; m++) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      mtx.lock();
      if (d < minDist[m]) {
        minDist[m] = d;
        args->clusterAssignments[m] = k;
      }
      mtx.unlock();
    }
}

void computeAssignmentsThreads (WorkerArgs &args){
  std::thread workThread[maxThreads];
  WorkerArgs threadArg[maxThreads];
  double *minDist=new double[args.M];
  int step=args.M/args.numThreads;
  std::mutex mtx;

  for (int m = 0; m < args.M; m++) {
    minDist[m] = 1e30;
    args.clusterAssignments[m] = -1;
  }
  for(int i=0;i<args.K;i++){
    threadArg[i]=args;
    threadArg[i].threadId=i;
  }
  for(int i=0;i<args.K;i++){
    workThread[i]=std::thread(computeAssignments,&threadArg[i],minDist,ref(mtx));
  }
  for(int i=0;i<args.K;i++){
    workThread[i].join();
  }
  free(minDist);
  return ;
}
```

再次编译运行：

![c8a6b81132822ecf83e7074c56f618e](.\graphs\c8a6b81132822ecf83e7074c56f618e.png)

总耗时缩短到8s，和之前的10s相比优化效果并不明显，一个原因是最后只有3个聚类中心，线程数不够多，不能完全发挥多线程的优势。另外频繁地加锁解锁导致线程阻塞，影响性能。
