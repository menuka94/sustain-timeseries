import datetime

f = open("profiler.out", "a")


def write_to_file(log):
    log_line = f'{datetime.datetime.now()}: {log}'
    print(log_line)
    f.write(log_line + '\n')
    f.flush()
