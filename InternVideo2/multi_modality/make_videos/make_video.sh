#!/bin/bash

# 进入目录 A（你可以修改为实际路径）
cd /path_to_images || exit

# 遍历 A 目录下的所有文件夹
for dir in */; do
  echo "处理文件夹: $dir"
  cd "$dir" || continue

  folder_name="${dir%/}"
  images=($(ls img_*.jpg | sort))
  count=${#images[@]}

  echo "发现 $count 张图片"

  if (( count % 2 == 1 )); then
    echo "图片数是奇数，重复最后一张图片"
    last_image="${images[-1]}"
    images+=("$last_image")
  fi

  total=${#images[@]}
  video_name="${folder_name}_0.0_${total}.0.mp4"

  # 创建文件列表
  rm -f filelist.txt
  for img in "${images[@]}"; do
    echo "file '$PWD/$img'" >> filelist.txt
  done

  # 创建视频
  # ffmpeg -y -f concat -safe 0 -r 1 -i filelist.txt -vsync vfr -pix_fmt yuv420p "../$video_name"
  # ffmpeg -r 1 -f concat -safe 0 -i filelist.txt -c:v libx265 -pix_fmt yuv420p "../$video_name"
  ffmpeg -y -f concat -safe 0 -r 1 -i filelist.txt \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libx264 -crf 23 -pix_fmt yuv420p "../$video_name"

  # rm -f filelist.txt
  cd ..
done
