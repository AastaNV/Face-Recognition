deploy_fd = '/home/vyu/Face/JEP/script/detection.prototxt'
deploy_fr = '/home/vyu/Face/JEP/script/classification.prototxt'
deploy_merge = '/home/vyu/Face/JEP/script/deploy.prototxt'

fp1 = open(deploy_fd, 'r')
fp2 = open(deploy_fr, 'r')
fp3 = open(deploy_merge, 'w')

line1 = fp1.readlines()
line2 = fp2.readlines()

for l in line1:
    tmp = l.replace(' ','')
    field = tmp.split(':')
    if( field[0]=='name' or field[0]=='top' or field[0]=='bottom'):
        source = field[1].split('"')[1]
        l = l.replace(source,(source+'_fd'))
        print 'proto replace: ' + source
    fp3.write(l)

for l in line2:
    tmp = l.replace(' ','')
    field = tmp.split(':')
    if( field[0]=='name' or field[0]=='top' or field[0]=='bottom'):
        source = field[1].split('"')[1]
        l = l.replace(source,(source+'_fr'))
        print 'proto replace: ' + source
    fp3.write(l)

fp1.close()
fp2.close()
fp3.close()
