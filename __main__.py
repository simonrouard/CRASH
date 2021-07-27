import os

from torch.cuda import device_count
from torch.multiprocessing import spawn

from learner import train, train_distributed

from params import params


def _get_free_port():
    import socketserver

    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]


def main():
    replica_count = device_count()
    if replica_count > 1:
        if params['batch_size'] % replica_count != 0:
            raise ValueError(
                f"Batch size {params['batch_size']} is not evenly divisble by # GPUs {replica_count}."
            )
        params['batch_size'] = params['batch_size'] // replica_count
        port = _get_free_port()
        spawn(
            train_distributed,
            args=(replica_count, port, params),
            nprocs=replica_count,
            join=True,
        )
    else:
        train(params)


if __name__ == "__main__":
    main()
