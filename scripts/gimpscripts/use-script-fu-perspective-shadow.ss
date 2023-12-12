;; 输入图像、输出图像、角度、水平的相对距离、阴影的相对长度、模糊半径、颜色、不透明度、插值、允许改变大小
(define (add-perspective-shadow infile outfile v-angle x-distance shadow-length blur color opacity toggle allow-update-size gradient-type bg-color gradient_strength)
  (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE infile infile)))
          (drawable (car (gimp-image-get-active-layer image))))
    ;; 图像、图层、角度、水平的相对距离、阴影的相对长度、模糊半径、颜色、不透明度、插值、允许改变大小
    (script-fu-perspective-shadow image drawable v-angle x-distance shadow-length blur color opacity toggle allow-update-size)
    (let* ((shadow-layer (car (gimp-image-get-layer-by-name image "Perspective Shadow")))
            (width (car (gimp-drawable-width shadow-layer)))
            (height (car (gimp-drawable-height shadow-layer)))
            )

      (gimp-context-set-gradient gradient-type)
;      (define gradient-name (car (gimp-context-get-gradient)))
;      (define new-gradient-name (car (gimp-gradient-duplicate gradient-name)))

;      (gimp-context-set-gradient new-gradient-name)

      ;; 获取渐变的段数
;      (define num-segments (car (gimp-gradient-get-number-of-segments new-gradient-name)))
      ;; 遍历每个段
;      (do ((i 0 (+ i 1)))
;        ((>= i num-segments))
;        ;; 设置段的左侧颜色为红色
;        (gimp-gradient-segment-set-left-color new-gradient-name i (list 0 0 0) 0)
;        ;; 设置段的右侧颜色为透明色
;        (gimp-gradient-segment-set-right-color new-gradient-name i bg-color 100))
      (do ((i 1 (+ i 1)))
        ((> i gradient_strength))
        (if (< v-angle 135)
          ;            (gimp-drawable-edit-gradient-fill shadow-layer 0 0 FALSE 1 1 FALSE 0 height width 0)
          (gimp-drawable-edit-gradient-fill shadow-layer 10 200 FALSE 1 1 TRUE 0 height (+ width 100) 0)
          (gimp-drawable-edit-gradient-fill shadow-layer 10 200 FALSE 1 1 TRUE width 0 -100 height)
          )
        )

      (gimp-file-save RUN-NONINTERACTIVE image shadow-layer outfile outfile)
      (let* ((merged (car (gimp-image-merge-visible-layers image CLIP-TO-IMAGE))))
        (gimp-file-save RUN-NONINTERACTIVE image merged outfile outfile)
        merged
        )
      )
    (gimp-image-delete image)))