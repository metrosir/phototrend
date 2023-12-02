(define (add-shadow infile outfile x y blur color opacity toggle )
 (display "add-shadow called")
 (newline)
 (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE infile infile)))
        (drawable (car (gimp-image-get-active-layer image))))
   ;;(script-fu-drop-shadow image drawable 10 10 10 '(0 0 0))
   (script-fu-drop-shadow image drawable x y blur color opacity toggle)
   (gimp-file-save RUN-NONINTERACTIVE image drawable outfile outfile)
   ; 合并图层
   (let* ((merged (car (gimp-image-merge-visible-layers image CLIP-TO-IMAGE))))
     (gimp-file-save RUN-NONINTERACTIVE image merged outfile outfile))
;   (do ((i 0 (+ i 1)))  ; 初始化循环变量i为0
;       ((= i 1))  ; 当i等于5时，结束循环
     ;; 图像、图层、偏移量x、偏移量y、模糊半径、颜色、允许调整大小和模式
;     (script-fu-drop-shadow image drawable 10 10 10 '(0 0 0) 50 100))  ; 调用script-fu-drop-shadow函数
;   (let* ((merged (car (gimp-image-merge-visible-layers image CLIP-TO-IMAGE))))
;     (gimp-file-save RUN-NONINTERACTIVE image merged outfile outfile))
   (gimp-image-delete image)))

(script-fu-register "add-shadow"
                  "Add Shadow"
                  "Add shadow to image"
                  "Your Name"
                  "Your Name"
                  "2023"
                  ""
                  SF-FILENAME "Input file" "/data1/aigc/gimp_test/input/0.png"
                  SF-FILENAME "Output file" "/data1/aigc/gimp_test/output/0.png")

(script-fu-menu-register "add-shadow" "<Image>/Filters")

;gimp -i -b '(add-shadow "/data1/aigc/gimp_test/input/deng.png" "/data1/aigc/gimp_test/output/deng.png" 50 10 20 "\'(0 0 0)" 50 0)' -b '(gimp-quit 0)'
; use-gimp-drawable-shadows-highlights.scm
;gimp -i -b '(add-shadow "/data1/aigc/phototrend/worker_data/history/simple_color_commodity/2023-12-03 00:31:22.png" "/data1/aigc/phototrend/worker_data/history/simple_color_commodity/2023-12-03 00:34:50.png" 50 10 20 "#000000" 50 0)' -b '(gimp-quit 0)'
;gimp -i -b '(add-shadow "/data1/aigc/phototrend/worker_data/history/simple_color_commodity/2023-12-03 00:31:22.png" "/data1/aigc/phototrend/worker_data/history/simple_color_commodity/2023-12-03 00:34:52.png" 50 10 20 "#000000" 1.23 0)' -b '(gimp-quit 0)'
;gimp -i -b '(add-shadow "/tmp/gradio/image.png" "/tmp/gradio/633398b0bd5fe317136d194b58751/image.png" "10" "10" "10" "#000000" "0.5" "0")' -b '(gimp-quit 0)'
;gimp -i -b '(add-shadow "/data1/aigc/gimp_test/input/deng.png" "/data1/aigc/gimp_test/output/deng.png")' -b '(gimp-quit 0)'
