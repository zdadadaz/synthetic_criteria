# from tensorflow.python.summary import event_accumulator as ea
from tensorboard.backend.event_processing import event_accumulator as ea
import matplotlib.pyplot as plt


def read_tb():
    path = "crossEncoder/models/3b/tc_med_model_63_3b_ance/runs/Dec05_22-57-42_gpunode-1-1/events.out.tfevents.1638709062.gpunode-1-1.272026.0"
    # path = 'crossEncoder/models/t5base/medt5_ps_model/runs/Dec05_21-10-45_gpunode-1-7/events.out.tfevents.1638702649.gpunode-1-7.156798.0'
    acc = ea.EventAccumulator(path)
    acc.Reload()

    # Print tags of contained entities, use these names to retrieve entities as below
    print(acc.Tags())

    # E. g. get all values and steps of a scalar called 'l2_loss'
    x = [s.step for s in acc.Scalars('train/loss')]
    y = [s.value for s in acc.Scalars('train/loss')]

    plt.plot(x, y)
    print(y)
    plt.savefig('/'.join(path.split('/')[:-1]) + '/tloss.png')


if __name__ == '__main__':
    read_tb()
