Font files for Frameit MVP
==========================

Place the following free Google Fonts TTF files in this directory:

  Caveat-Regular.ttf
    Download: https://fonts.google.com/specimen/Caveat
    Used for: caption zones, text label decorations

  CourierPrime-Regular.ttf
    Download: https://fonts.google.com/specimen/Courier+Prime
    Used for: film strip frame numbers

The renderer falls back to PIL's built-in bitmap font if these files
are absent. Captions will still render but look basic.

Quick download with curl:
  curl -L 'https://fonts.gstatic.com/s/caveat/v18/WnznHAc5bAfYB2QRah7pcpNvOx-pjfJ9eIWpZA.woff2'    -- (woff2 only; for TTF visit the Google Fonts page above)
