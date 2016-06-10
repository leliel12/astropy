from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import download_file

fits_file = 'http://data.astropy.org/tutorials/FITS-images/HorseHead.fits'
image_file = download_file(fits_file, cache=True)
hdu = fits.open(image_file)[0]
wcs = WCS(hdu.header)

fig = plt.figure()
fig.add_subplot(111, projection=wcs)
plt.imshow(hdu.data, origin='lower', cmap='cubehelix')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.show()