import progressbar as ProgressBar


def create_pbar(max_val, name):
    widgets = [name +': ', ProgressBar.Percentage(), ' ', ProgressBar.Bar(marker='#',left='[',right=']'),
        ' ', ProgressBar.ETA()] #see docs for other options

    pbar = ProgressBar.ProgressBar(widgets=widgets, maxval=max_val)
    pbar.start()
    return pbar

def matrix_string(matrix):
    return '\t['+']\n\t['.join('\t'.join('%0.3f' %x for x in y) for y in matrix) + ']'