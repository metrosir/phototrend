(define (add-shadow-v2 infile outfile)
 (display "add-shadow-v2 called")
 (newline)
 (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE infile infile)))
        (drawable (car (gimp-image-get-active-layer image))))
   (let* ((new-drawable (gimp-drawable-shadows-highlights drawable 20 -20 0 10 0 0 0)))
     (gimp-file-save RUN-NONINTERACTIVE image new-drawable outfile outfile))
   (gimp-image-delete image)))

(script-fu-register "add-shadow-v2"
                  "Add Shadow"
                  "Add shadow to image"
                  "Your Name"
                  "Your Name"
                  "2023"
                  ""
                  SF-FILENAME "Input file" "/data1/aigc/gimp_test/input/0.png"
                  SF-FILENAME "Output file" "/data1/aigc/gimp_test/output/0.png")

(script-fu-menu-register "add-shadow-v2" "<Image>/Filters")

; gimp -i -b '(add-shadow-v2 "/data1/aigc/gimp_test/input/deng.png" "/data1/aigc/gimp_test/output/deng.png")' -b '(gimp-quit 0)'