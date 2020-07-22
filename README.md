# python-superxBR
A python application for integer scaling pixel art to higher resolutions using Hyllian's Super-xBR algorithm.

Adapted from Hyllian's C++ version of Super-xBR - https://pastebin.com/cbH8ZQQT

# Execution

## From the command line:

>python py-superxbr.py input.png output.png 2

## From another Python script:

>import python-superxbr
>python-xbr.scale(input.png, output.png, 2)


# Examples

| Original image        | Scaled 2x (same size) |
| ------------- |-------------|
| <img src=https://i.imgur.com/GLFpO76.png height="192"/> | <img src=https://i.imgur.com/WMktwGw.png height="192"/> |
| <img src=https://i.imgur.com/WN535Hm.png height="240"/> | <img src=https://i.imgur.com/6XkhkLt.png height="240"/> |

| Original image        | Scaled 2x (full size) |
| ------------- |-------------|
| <img src=https://i.imgur.com/GLFpO76.png height="192"/> | <img src=https://i.imgur.com/WMktwGw.png/> |
| <img src=https://i.imgur.com/WN535Hm.png height="240"/> | <img src=https://i.imgur.com/6XkhkLt.png/> |

# License

This project is licensed under the MIT License. See LICENSE.md for more detail.

# Acknowledgments

Hyllian for their Super-xBR algorithm and reference PDF/C++ code.

GIMP-superxBR, a plug-in for GIMP which was the basis for this application. By [abelbriggs1](https://github.com/abelbriggs1/gimp-superxBR)
