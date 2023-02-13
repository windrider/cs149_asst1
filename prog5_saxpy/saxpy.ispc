
export void saxpy_ispc(uniform int N,
                       uniform float scale,
                            uniform float X[],
                            uniform float Y[],
                            uniform float result[])
{
    foreach (i = 0 ... N) {           
        result[i] = scale * X[i] + Y[i];
    }
}

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
        //result[i] = scale * X[i];
    }
    /*
    foreach (i = indexStart ... indexEnd) {
        //result[i] = scale * X[i] + Y[i];
        result[i] += Y[i];
    }
    */
}

/*
task void saxpy_ispc_task_2(uniform int N,
                               uniform int span,
                               uniform float scale,
                               uniform float X[], 
                               uniform float Y[],
                               uniform float result[])
{
    uniform int indexStart = taskIndex * span;
    uniform int indexEnd = min(N, indexStart + span);

    foreach (i = indexStart ... indexEnd) {
        result[i] += Y[i];
    }
}
*/

export void saxpy_ispc_withtasks(uniform int N,
                               uniform float scale,
                               uniform float X[],
                               uniform float Y[],
                               uniform float result[])
{

    uniform int span = N / 2;  // 64 tasks

    launch[N/span] saxpy_ispc_task(N, span, scale, X, Y, result);
    //launch[N/span] saxpy_ispc_task_2(N, span, scale, X, Y, result);
}
