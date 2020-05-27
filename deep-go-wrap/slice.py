import numpy as np
import h5py
import sys

dset = h5py.File(sys.argv[1], 'r')
sliceChunk = (dset['xs'].shape[0] / 100000) + 1

for i in range(sliceChunk) :
    print('slicing hdf5[%s]: %s of %s' % (sys.argv[1], sliceChunk - 1, i))
    if i == sliceChunk - 1:
        xs = dset['xs'][(i*100000):(dset['xs'].shape[0])]
        ys = dset['ys'][(i*100000):(dset['xs'].shape[0])]
    else:
        xs = dset['xs'][(i*100000):((i+1)*100000)]
        ys = dset['ys'][(i*100000):((i+1)*100000)]
    targ = h5py.File(str('%s_%s.hdf5' % (sys.argv[1][:-5], i)),'w')
    targ.create_dataset('xs', data=xs, compression="gzip")
    targ.create_dataset('ys', data=ys, dtype='u2', compression="gzip")
    del xs
    del ys
    targ.close()
    del targ
    print('sliced hdf5[%s] dataset %s saved.' % (sys.argv[1], str('%s_%s.hdf5' % (sys.argv[1][:-5], i))))
