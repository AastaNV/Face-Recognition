import sys 
import caffe

deploy_fd = 'detection.prototxt'
deploy_fr = 'classification.prototxt'
deploy_merge = 'deploy.prototxt'

model_fd = 'detection.caffemodel'
model_fr = 'classification.caffemodel'
model_merge = 'snapshot_iter_1.caffemodel'

net_fd = caffe.Net(deploy_fd,model_fd, caffe.TEST)
net_fr = caffe.Net(deploy_fr,model_fr, caffe.TEST)
net_merge = caffe.Net(deploy_merge,model_merge, caffe.TEST)

fp1 = open(deploy_fd, 'r')
fp2 = open(deploy_fr, 'r')
line1 = fp1.readlines()
line2 = fp2.readlines()

for l in line1:
    tmp = l.replace(' ','')
    field = tmp.split(':')
    if( field[0]=='name'):
        source = field[1].split('"')[1]
        target = source+'_fd'
        try:
            for i in range(len(net_fd.params[source])):
                net_merge.params[target][i].data[...] = net_fd.params[source][i].data[...]
            print 'update weight: ' + target
        except KeyError:
            print 'ignore weight: ' + target

for l in line2:
    tmp = l.replace(' ','')
    field = tmp.split(':')
    if( field[0]=='name'):
        source = field[1].split('"')[1]
        target = source+'_fr'
        try:
            for i in range(len(net_fr.params[source])):
                net_merge.params[target][i].data[...] = net_fr.params[source][i].data[...]
            print 'update weight: ' + target
        except KeyError:
            print 'ignore weight: ' + target

fp1.close()
fp2.close()

net_merge.save('merge.caffemodel')
