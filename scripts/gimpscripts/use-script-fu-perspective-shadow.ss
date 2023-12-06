;; 输入图像、输出图像、角度、水平的相对距离、阴影的相对长度、模糊半径、颜色、不透明度、插值、允许改变大小
(define (add-perspective-shadow infile outfile v-angle x-distance shadow-length blur color opacity toggle allow-update-size)
 (display "add-shadow called")
 (newline)
 (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE infile infile)))
       (drawable (car (gimp-image-get-active-layer image))))
  ;; 图像、图层、角度、水平的相对距离、阴影的相对长度、模糊半径、颜色、不透明度、插值、允许改变大小
  (script-fu-perspective-shadow image drawable v-angle x-distance shadow-length blur color opacity toggle allow-update-size)
  (gimp-file-save RUN-NONINTERACTIVE image drawable outfile outfile)
  ; 合并图层
  (let* ((merged (car (gimp-image-merge-visible-layers image CLIP-TO-IMAGE))))
    (gimp-file-save RUN-NONINTERACTIVE image merged outfile outfile)
    merged)
  (gimp-image-delete image)))


;gimp -i -b '(script-fu-perspective-shadow "/data1/aigc/phototrend/worker_data/history/simple_color_commodity/2023-12-03 00:31:22.png" "/data1/aigc/phototrend/worker_data/history/simple_color_commodity/11.png" 39 6 0.3 30 "#000000" 23 0 0)' -b '(gimp-quit 0)'
;gimp -i -b '(add-perspective-shadow "/data1/aigc/phototrend/worker_data/history/simple_color_commodity/2023-12-03 00:31:22.png" "/data1/aigc/phototrend/worker_data/history/simple_color_commodity/11.png" 39 6 0.3 30 "#000000" 23 0 0)' -b '(gimp-quit 0)'