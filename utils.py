import time

CUR_TIME = time.time()

def time_log(job, start_time=0):
    
    global CUR_TIME
    
    if start_time != 0:
        print('\n%s time: %.6f sec' % (job, time.time() - start_time))    
    else:
        print('\n%s time: %.6f sec' % (job, time.time() - CUR_TIME))
    
    CUR_TIME = time.time()

    print()