from read_file import Data
from receiver.receiver import Receiver
from transmitter.transmitter import Transmitter

    
if __name__=='__main__':
    data = Data().data

    transmitter = Transmitter(data).compress()

    receiver = Receiver()
