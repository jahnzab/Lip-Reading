import gdown
url = 'https://drive.google.com/uc?id=1_H6KrQAGBu4vl2i3_wsDq0xztOOjI7hK&confirm=t'
#url='https://drive.google.com/file/d/1_H6KrQAGBu4vl2i3_wsDq0xztOOjI7hK/view?usp=drive_link'
#url= 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
output = 'data.zip'
gdown.download(url, output, quiet=False)
