import numpy as np

def get_metrics(CM):
    tp = CM[0,0]
    tn = CM[1,1]
    fp = CM[1,0]
    fn = CM[0,1]

    # precision, recall, and accuracy
    precision = tp/(tp + fp)
    recall = tp/(tp+fn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    print(f'accuracy: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}')

    # matthew's correlation coefficient
    mcc = (tn*tp - fn*fp)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    print(f'mcc: {mcc:.3f}')

def evaluate(CM):
    print(f'{CM[0,0]+CM[0,1]} in class A, {CM[0,0]} correctly predicted,')
    print(f'{CM[1,0]+CM[1,1]} in class B, {CM[1,1]} correctly predicted')
    get_metrics(CM)


def test(CM, num):
    print(f'>>>>> test {num} >>>>>>')
    print(CM)
    evaluate(CM)
    
    print('\n------swapping labels--------')
    # switch labels by swapping rows, then swapping columns
    CM = CM[[1,0]][:,[1,0]]
    evaluate(CM)
    print('\n\n')


# create test cases of confusions matrices
cms = [
    np.array([[999, 1],
            [9,1]]),
    np.array([[999, 1],
            [1,9]]),
    np.array([[999, 1],
            [998, 2]]),
    np.array([[1, 999],
            [9, 1]]),
    np.array([[999, 1],
            [2, 998]])
]    

for i, cm in enumerate(cms):
    test(cm, i+1)
        
        
        