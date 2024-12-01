---
# 文章标题
title: Project Zombie源码分析
# 发布日期
date: 2024-05-28
# 文章描述（副标题）
description: 不过还是很菜什么都写不出来

# 文章底部显示的关于内容的标签
tags: 
    - Project Zombie
# 是否启用对KaTeX的支持，默认为false
math: false
    
# 顶部显示的分类标签
categories:
    - 源码分析
---
 **使用游戏版本：41.78.16**

## 游戏目录文件结构

ProjectZomboid/ \
├── media/ \
│   ├── actiongroups/ \
│   │   └── (包含描述动画过渡的 .xml 脚本) \
│   ├── anims_X/ \
│   │   └── (包含动画文件) \
│   ├── AnimSets/ \
│   │   └── (包含描述动画参数的 .xml 脚本) \
│   ├── lua/ \
│   │   └── (包含游戏的所有 Lua 代码) \
│   ├── maps/ \
│   │   └── (包含地图文件) \
│   ├── models_X/ \
│   │   └── (包含物品、武器和衣服的 3D 模型) \
│   ├── radio/ \
│   │   └── (包含描述电台操作的脚本) \
│   ├── scripts/ \
│   │   └── (包含描述物品、武器、车辆参数的脚本) \
│   ├── sound/ \
│   │   └── (包含游戏的音频文件) \
│   ├── texturepacks/ \
│   │   └── (包含打包的游戏纹理和图像) \
│   ├── textures/ \
│   │   └── (包含游戏未打包的纹理) \
│   └── ui/ \
│       └── (包含游戏的图标和图像) \
└── zombie/ \
&nbsp;&nbsp;&nbsp;&nbsp;└── (包含游戏的编译 Java 代码)

## 用户目录文件结构
在Windows系统中，用户目录的路径通常为：`C:/Users/username/Zomboid`

Zomboid/（游戏根目录）\
├── console.txt（控制台日志文件）\
├── coop-console.txt（本地主机服务器日志文件）\
├── server-console.txt（专用服务器日志文件）\
├── Lua/（Lua脚本文件夹）\
├── mods/（本地存储的mod文件夹）\
├── Workshop/（Steam创意工坊文件夹）\
└── Saves/（游戏存档文件夹）


## 模组文件结构

mod根目录/\
├── mod.info （包含mod及相关内容的描述）\
├── image.png （mod的封面图片，供mod.info引用）\
└── media/\
&nbsp;&nbsp;&nbsp;&nbsp;├── models_X/\
&nbsp;&nbsp;&nbsp;&nbsp;│   └── (3D 模型文件：.x, .fbx)\
&nbsp;&nbsp;&nbsp;&nbsp;├── scripts/\
&nbsp;&nbsp;&nbsp;&nbsp;│   └── (脚本文件：物品、车辆、配方、衣服)\
&nbsp;&nbsp;&nbsp;&nbsp;├── textures/\
&nbsp;&nbsp;&nbsp;&nbsp;│   └── (纹理图像文件：物品、衣服、车辆)\
&nbsp;&nbsp;&nbsp;&nbsp;├── ui/\
&nbsp;&nbsp;&nbsp;&nbsp;│   └── (图标、界面图像文件)\
&nbsp;&nbsp;&nbsp;&nbsp;├── texturepacks/\
&nbsp;&nbsp;&nbsp;&nbsp;│   └── (纹理包文件)\
&nbsp;&nbsp;&nbsp;&nbsp;├── lua/\
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── shared/\
&nbsp;&nbsp;&nbsp;&nbsp;│   │   └── (服务器端和客户端共用的 Lua 文件)\
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── client/\
&nbsp;&nbsp;&nbsp;&nbsp;│   │   └── (仅在客户端执行的 Lua 文件)\
&nbsp;&nbsp;&nbsp;&nbsp;│   └── server/\
&nbsp;&nbsp;&nbsp;&nbsp;│       └── (仅在服务器端执行的 Lua 文件)\
&nbsp;&nbsp;&nbsp;&nbsp;├── sound/\
&nbsp;&nbsp;&nbsp;&nbsp;│   └── (音频文件)\
&nbsp;&nbsp;&nbsp;&nbsp;├── maps/\
&nbsp;&nbsp;&nbsp;&nbsp;│   └── (地图文件)\
&nbsp;&nbsp;&nbsp;&nbsp;├── animSets/\
&nbsp;&nbsp;&nbsp;&nbsp;│   └── (描述动画参数的 XML 脚本)\
&nbsp;&nbsp;&nbsp;&nbsp;└── anims_X/\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── (动画文件：.x, .fbx)\


## 游戏代码结构
PZ同时使用 Java 和 Lua。主要引擎和 API 功能使用 Java，大部分逻辑使用 Lua 脚本，便于在不需要编译的情况下进行修改。

为了将Lua嵌入到Java中，PZ使用了`se.krka.kahlua` 库，这是一个纯Java编写的，轻量级Lua虚拟机，可以在java中解释执行Lua脚本，也可以在Lua中调用Java的类和方法。相比于`LuaJ`库可以把Lua编译成Java字节码，`kahlua`是解释执行Lua，性能相对较差。

Lua源代码位于 `ProjectZomboid/media/lua` 中，是完全公开的，包含三个子文件夹，用于客户端的 `client` 、用于服务器端的 `server` 以及两端共用的 `shared` ，**有些lua文件包含一些开发者的注释**。

Java源代码位于 `ProjectZomboid/Zombie` 中，因为是编译过的`.class`文件，所以通过反编译才能阅读源代码。

Java源代码中的部分API已经通过Lua公开，具体方式是，在lua文件中先声明一个全局的空表，然后在这个表中定义与Java API同名的**空函数**，**公开的Java API都被写入了Lua文件的全局命名空间（表）中**。

这意味着在`kahlua`库的帮助下，在Lua中调用同名的Lua全局函数，实际上是间接调用Java中的类和方法，不需要任何额外操作，在任何lua文件中都可以调用，不同的是lua中使用冒号`:`代替Java中的`.`运算符访问成员变量和方法。

如果要了解这些函数做了什么，就不得不阅读反编译的Java代码。

因为lua中经常调用Java API，所以很可能返回一些Java对象，如`ArrayList`等，在lua中也要通过Java的方法来访问，例如：
```lua
arrayList:get(0)
```
实际上，由于lua嵌入Java执行的需要，**Java的很多原生方法在Lua中也是全局可用的**，但这是因为开发者把它们写进了Lua的全局命名空间以及`kahlua`库的作用，而不是lua本身的特性。

虽然Java拥有很多方便好用的方法，需要注意的是，**通过lua间接调用Java会有一定的性能损失**。

## Lua源代码解析
### lua/shared
```lua
VehicleZoneDistribution.lua：更改或添加车辆生成逻辑
SpawnRegions.lua：加载用户目录中的出生点文件“servertest_spawnregions.lua”
luautils.lua：定义了一系列实用的lua工具函数，包括仿Java的字符串处理函数，以及一些游戏内的与玩家、物品、方格、物品栏、耐久度等相关的操作函数。
keyBinding.lua：定义了游戏中键盘按键与功能的映射。
ISBaseObject.lua：定义了基本对象类，可创建对象，type属性默认为“ISBaseObject”，也可以传入一个字符串用于更改类型。
defines.lua：定义了一个全局表，包含很多物品掉落和玩家状态参数的键值对。
BodyLocations.lua：针对人物的各种服饰等，创建身体位置。各种衣物、包括绷带、伤口、甚至僵尸伤害都会占用身体位置，有些东西可以在其位置上存在多个（绷带、伤口、僵尸伤害）。还定义了一些衣物的互斥规则（不能同时穿戴）。

```
