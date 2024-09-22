import os
import argparse
import shutil

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def splitMSEC(input,target):
    assert os.path.exists(input), 'Input dir not found'

    Exp1 = os.path.join(target,'input', 'Exp1')
    Exp2 = os.path.join(target,'input', 'Exp2')
    Exp3 = os.path.join(target,'input', 'Exp3')
    Exp4 = os.path.join(target,'input', 'Exp4')
    Exp5 = os.path.join(target,'input', 'Exp5')

    os.makedirs(Exp1, exist_ok=True)
    os.makedirs(Exp2, exist_ok=True)
    os.makedirs(Exp3, exist_ok=True)
    os.makedirs(Exp4, exist_ok=True)
    os.makedirs(Exp5, exist_ok=True)

    imgs = os.listdir(input)
    for img in imgs:

        a = img.rfind('_')
        imgname = img[:a]+'.jpg'
        flag = img[a:-4]

        if flag =='_0':
            shutil.copy(os.path.join(input, img), os.path.join(Exp1, imgname))
        elif flag == '_N1.5':
            shutil.copy(os.path.join(input, img), os.path.join(Exp2, imgname))
        elif flag == '_N1':
            shutil.copy(os.path.join(input, img), os.path.join(Exp3, imgname))
        elif flag == '_P1.5':
            shutil.copy(os.path.join(input, img), os.path.join(Exp4, imgname))
        elif flag == '_P1':
            shutil.copy(os.path.join(input, img), os.path.join(Exp5, imgname))

    print('total Exp1:', len(os.listdir(Exp1)))
    print('total Exp2:', len(os.listdir(Exp2)))
    print('total Exp3:', len(os.listdir(Exp3)))
    print('total Exp4:', len(os.listdir(Exp4)))
    print('total Exp5:', len(os.listdir(Exp5)))



def writePath(inputdir,outputdir,targetdir):
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    mkdir(outputdir)
    Exp = ['Exp1','Exp2','Exp3','Exp4','Exp5']
    for item in Exp:
        inputdir = os.path.join(inputdir, item)
        imgs = os.listdir(inputdir)
        for img in imgs:

            targetimg = img[:-4]+'.jpg'

            groups = ''

            groups += os.path.join(inputdir, img) + '|'
            groups += os.path.join(targetdir,targetimg)

            with open(os.path.join(outputdir, 'test_1.txt'), 'a') as f:
                f.write(groups + '\n')
        inputdir = os.path.dirname(inputdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/ubuntu/home/qgf/Exposure/test/input', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--target', type=str, default='/home/ubuntu/home/qgf/Exposure/test/target', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='./data/', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    writePath(inputdir,outputdir,targetdir)

    #splitMSEC('E:/Datesets/MSEC/testing/INPUT_IMAGES','E:/Datesets/Exposure/test')

