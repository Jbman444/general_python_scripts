import qrcode
data = 'https://www.moraleslab.ucdavis.edu/'

img = qrcode.make(data)
img.save('/home/jorge/Documents/scripts/Morales_lab_qr.png')