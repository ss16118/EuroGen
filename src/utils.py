import sys


# Custom progress bar
def progress_bar(curr_progress, total, label, auxiliary_info):
    sys.stdout.write('\r')
    percentage_finished = curr_progress / float(total)
    sys.stdout.write("[%-40s] %d%% | %d/%d | %s: %s" %
                     ('=' * int(percentage_finished * 40),
                      100 * percentage_finished, curr_progress, total, label, auxiliary_info))
    sys.stdout.flush()
