pro split, inpfile, outdir

; The inpfile must be of the format [x, y, 90, number of seconds (one r0 for each second)] for mihi data

  im = double(ftsrd(inpfile,0,h))
  print, 'Image read: ', size(im)
  for i=30L,134L do begin
    print, i
    for k=0L,9L do begin
      for j=0L,89L do begin
        filename = '/image.' + string(57600 + i*900L + k*90 + j, format='(I09)') + '.f0'
        fzwrite,fix(reform(im[*,*,j,i*10+k])), outdir + filename, '0'
      endfor
    endfor
  endfor
end
