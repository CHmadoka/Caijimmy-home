---
# 文章标题
title: Lua基础
# 发布日期
date: 2024-05-24
# 文章描述（副标题）
description: 轻量级脚本语言
# 文章底部显示的关于内容的标签
tags: 
    - 脚本
    - 性能优化
# 是否启用对KaTeX的支持，默认为false
math: false
image: logo.png
    
# 顶部显示的分类标签
categories:
    - 编程语言
    - Lua
---


## 变量与数据类型
### 声明变量
```lua
local a = 10  -- 局部变量
b = 20        -- 全局变量
```

### 数据类型
```lua
nil: 表示一个空值或未定义（在脚本语言中，变量为null等同于undefined）
boolean: 包含两个值：true 和 false
number: 双精度浮点数（lua中唯一的数值类型）
string: 字符串
table: 表，Lua 唯一的数据结构
function: 函数
userdata: 用户自定义类型
thread: 线程
```

## 控制结构
### 条件语句
```lua
local x = 10
-- Lua中，不等号为 ~=
if x~=10 then
    return
end

if x > 5 then
    print("x is greater than 5")
elseif x == 5 then
    print("x is equal to 5")
else
    print("x is less than 5")
end
```
### 循环
```lua
-- while 循环
local i = 1
while i <= 5 do
    print(i)
    i = i + 1
end

-- for 循环
for j = 1, 5 do
    print(j)
end

-- 泛型 for 循环
local t = {10, 20, 30}
for index, value in ipairs(t) do
    print(index, value)
end

-- do while循环(repeat until)
local i = 1
repeat 
    i = i + 1
until i>5
print(i)
```
使用 **break** 关键字可以跳出循环，但是Lua中并没有 continue 关键字可以直接进入下一次循环。

## 函数
```lua
-- 定义函数
function add(a, b)
    return a + b
end

-- 调用函数
local sum = add(3, 4)
print(sum)  -- 输出: 7

-- 匿名函数
-- lua中函数可以用变量存储，然后通过变量调用函数。
function_subtract = function(a, b)
    return a - b
end
print(function_subtract(10, 5))  -- 输出: 5
```

### 函数作用域
注意，lua中的函数和变量，包括函数中定义的变量，默认都是全局作用域，除非使用local关键字声明！

```lua
-- 定义局部函数
local function add(a, b)
    -- 函数中定义的变量，默认是全局变量！
    c=5
    return a + b + c
end

print(add(3,4)) -- 12
print(c) -- 5

```

>Lua 的核心理念之一是在尽可能最小的范围（本地）内声明变量和函数。\
\
所以定义变量和函数，以及在函数中定义变量时，尽可能使用local关键字。

如果函数中声明了一个变量，并且需要在函数外使用该变量，应该在函数定义前，现将其声明为局部变量：
```lua
local c
local function add(a, b)
    -- 函数中定义的变量，默认是全局变量！
    c=5
    return a + b + c
end
```


## 表（Tables）
表是lua中唯一的数据结构，但非常灵活，可以实现复杂多样的数据结构，如数组，字典等。

表中的元素使用逗号间隔。

表元素不能使用local关键字声明，表元素的作用域与表一致。

```lua
-- 定义表
local person = {
    nilValue = nil, -- nil 空值或未定义
    booleanValue = true, -- boolean
    numberValue = 42, -- number 双精度浮点数
    stringValue = "Hello, World!", -- string 字符串
    tableValue = {1, 2, 3}, -- table 表
    123，
    functionValue = function(x, y) return x + y end, -- function 函数
    userDataValue = userdata,  -- userdata（用户自定义类型）
    threadValue = coroutine.create(function() print("Coroutine") end),

    -- 表中可以同时包含数组元素和键值对
    "element",
    123.89,
    {1, "two", false}
}
```

注意，为了兼容字典和数组类型，表把数组元素的索引作为其键。

因此表可以分为两个部分，即数组部分和哈希部分。

表中的数组部分是顺序序列，可以通过索引访问。lua中，索引从1开始计数！！！表中的数组元素默认按照先后顺序排列。
可以通过 “[整数]=value” 的键值对形式，显式地指定数组元素的索引。

如：[5]="five"，指定表的数组部分第5个元素为字符串“five”。

哈希部分是键值对元素，只能通过点运算符直接访问。

```lua
-- 通过 #table ，可以获取表中的数组部分的长度。
print(#person)

-- 访问表的键值元素
print(person.numberValue)  -- 输出: 42

-- 插入新元素
-- table.insert()方法，接收3个参数，表、索引和值，用于给表的数组部分插入值。索引不能大于 数组部分的长度+1，如果省略索引，则在数组部分的末尾插入。
table.insert("key", 6, "six")
-- 通过点运算符赋值的方式，插入键值对。
person.gender = "male"


-- 遍历表
-- key, value in pairs() 获取表的所有键值对
for key, value in pairs(person) do
    print(key, value)
end

-- 遍历表的数组部分
for i = 1, #person do
    print(i, person[i])
end
```

### 表函数的定义
使用匿名函数
```lua
t={}
--表外添加键值对
t.func=function(a,b)
    return a+b
end

--表内定义
t={
    func=function(a,b)
        return a+b
    end
}
```

直接定义
```lua
t={}

function t:func(a,b)
    return a+b
end
```

### 元表与元方法
拥有元方法的表称为元表，原方法是一些拥有特定名称的函数。

元表允许改变表的行为，每个表都可以有一个元表，元表的元方法用于定义表的一些特定操作，如算术运算、索引、调用等。
```lua

-- 定义一个表
local t = {1, 2, 3}

-- 定义一个元表
local mt = {
    -- __add是特定的元方法，用于定义两个表相加时，执行的操作。
    __add = function(t1, t2)
        for i = 1, #t2 do
            table.insert(t1, t2[i])
        end
        return t1
    end
}

-- setmetatable函数，给表绑定元表
setmetatable(t, mt)

-- 使用元方法
local t2 = {4, 5, 6}
local t3 = t + t2

for _, value in ipairs(t3) do
    print(value)  -- 输出: 1 2 3 4 5 6
end
```

元方法及其作用如下：
```lua
__index:定义访问表中不存在的键时的行为。定义类时，通常使其等于类名（表名），否则无法通过对象名称访问成员方法。
__newindex:定义给表中不存在的键赋值时的行为。
__call:定义表作为函数调用时的行为。

__tostring:定义表转换成字符串时的值。
__len:定义对表使用操作符“#”时的行为。

-- 算术运算
__add:表相加
__sub:表相减
__mul:表相乘
__div:表相除

-- 逻辑判断(没有判断大于和大于等于的元方法)
__eq:==
__lt:<
__le:<=
```

## 面向对象编程
Lua 通过表和元表来实现面向对象编程。
```lua
-- 定义一个类
local Animal = {}
Animal.__index = Animal

-- 构造函数一般命名为new
function Animal:new(name)
    -- 一般先用setmetatable函数，给一个空表绑定元表，然后返回该空表，
    -- 并在其基础上设置表元素。
    local instance = setmetatable({}, Animal)
    instance.name = name
    return instance
end

-- 类的成员方法
function Animal:speak()
    print("My name is " .. self.name)
end

-- 创建实例
local dog = Animal:new("Buddy")
dog:speak()  -- 输出: My name is Buddy
```
值得注意的是，new方法返回的对象仅仅是包含表中变量的一个新表，而类的成员方法定义在元表中，所以无法直接通过该对象访问元表中定义的成员方法。

但由于对象与元表绑定，并且元表中定义了元方法 __index 为元表的名称，所以当通过该对象访问未定义的函数时，就会触发元方法，把调用对象替换成元表，从而实现通过对象调用类的成员方法。

## 字符串操作
```lua
local s = "Hello, World"
print(#s)  -- #输出字符串长度
print(string.upper(s))  -- 转换为大写
print(string.lower(s))  -- 转换为小写
```


## require函数
```lua
require("moduleName") --注意模块名称为文件名省略.lua后缀。
```
require 函数用于加载并运行一个模块。如果模块之前已经加载过，require 会返回该模块的缓存结果，而不会重复加载。**这意味着在同时引入该文件中的全局变量和函数**。

这个机制确保了模块在 Lua 脚本中只会被加载一次，避免了重复加载带来的性能问题和潜在的副作用。

## 模块
模块用于将功能相关的代码封装在一个独立的命名空间中，从而避免命名冲突，模块也是使lua脚本之间进行数据传递和代码引用的重要机制，模块可以返回任意数据类型。

定义模块时，使用点运算符创建该命名空间独有的变量，并且在lua文件中通过return语句返回模块。
每个lua脚本只能调用一次return语句（返回一个模块）

模块通常是局部变量，因为如果是全局变量，就不必在当前文件return，也不必在其他文件require，其它文件可以直接使用全局变量（前提是这些文件建立了联系）。
```lua
-- 定义模块(mymodule.lua)
local M = {} 
M.value = 0 --通过点运算符，在表M的命名空间中定义了变量value

function M.increment()
    M.value = M.value + 1
end
-- 返回模块
return M
```
使用模块时，需要在当前lua脚本中，调用require函数并传入模块所在的文件名（不包含.lua后缀）。require函数返回模块。

```lua
-- 使用模块(main.lua)
local mymodule = require("mymodule")
mymodule.increment()
print(mymodule.value)  -- 输出: 1
```

## 函数的覆盖与保留
有时候需要重写全局函数，同时能够调用原函数。可以先使用变量存储原函数，再重写原函数以覆盖：
```lua
-- ISToolTipInv.render是来自其他文件的表函数
local original_render = ISToolTipInv.render

function ISToolTipInv:render()
-- 可以根据条件判断，调用原函数，还是重写后的函数。
    if not CONDITION then
        original_render(self)
    end
    -- ... some custom code ...
end

function ISToolTipInv:render()
-- 先执行原函数，再执行新的步骤
    original_render(self)
-- 新的代码：

end
```

## 作用域 全局与局部
全局变量和函数可以从任何文件、代码中的任何位置访问。

局部变量只能从声明它们的文件或代码块访问。

与函数不同，在表中定义的变量或函数等，作用域与表一致。

虽然全局作用域是一个方便的功能，但它往往更成问题。访问全局变量速度较慢，而且存在被意外覆盖的风险。Lua 的核心理念之一是在尽可能最小的范围（本地）内声明变量和函数。
### 特殊规则
声明变量时，只要没有使用local关键字，就一定是全局变量。

声明函数时如果没有使用local关键字，一般是全局函数，除非该函数被封装在局部变量或局部表中：
```lua
local func=function()
    print("函数被封装在局部变量中，因此函数是局部函数。")
end

local tb={}
function tb:func()
    print("函数被封装在局部表中，因此函数是局部函数。")
end
```

这符合访问权限的控制规则和一致性。

### 跨文件访问全局变量
只需要在当前文件中引入目标文件，就可以直接访问目标文件中定义的全局变量（包含模块）。

### 跨文件访问局部变量
局部变量的作用域是其所在的文件或代码块，因此无法直接跨文件访问，只能通过lua的模块机制或全局函数实现这一点。

通过全局函数访问：
```lua
-- file1.lua
local localVar=1

function getLocalVar()
    return localVar
end
```

```lua
-- file2.lua
require ("file1") -- 引入file1.lua中的全局函数

print(getLocalVar())
```

通过模块访问：
```lua
-- file1.lua
local M={}

local M.localVar=1

-- 模块内定义一个get函数，隐藏内部细节，实现数据封装
function M.getLocalVar()
    return M.localVar
end

return M
```

```lua
-- file2.lua
local M=require("file1")

print(M.getLocalVar())

return 
```
虽然使用get和set方法有很多好处，但并非总是需要使用get和set方法访问模块或表内字段，尤其是需要频繁访问这些字段时，相比于直接访问字段，使用中间方法会降低性能。

## 性能优化
>代码块运行得越频繁，就越需要对其进行优化。

### 作用域
1. 尽可能使用local关键字声明变量和函数

2. 函数内部始终使用局部变量，如果函数外需要使用该局部变量，就现在函数外将其声明为局部变量。
```lua
local x
local function Fun()
    x = 5
    return "something else"
end
```
3. 把文件中所有的全局作用域函数和变量，放入一个全局表中，避免全局命名空间的污染，便于使用和管理。

4. 把需要使用的全局表和函数存到局部变量中。Lua的最小作用域理念不仅适用于声明变量和函数，还适用于访问它们。通过将全局变量拉入本地空间，可以提高性能。

```lua
local MyTable = {}
function MyTable.Fun(table_list)
    for _, num in ipairs(table_list)
        -- 每次循环都要调用全局函数print和ZombRand
        print(ZombRand(num))
    end
end
```
把全局变量引入到本地后：
```lua
local MyTable = {}
function MyTable.Fun(table_list) 
    local print = print
    local ZombRand = ZombRand
    for _, num in ipairs(table_list)
        print(ZombRand(num))
    end
end
```
更极端的做法是：
```lua
local MyTable = {}
local print = print
local ZombRand = ZombRand
local ipairs = ipairs -- 此时完全不需要在全局空间中查找
function MyTable.Fun(table_list) 
    for _, num in ipairs(table_list)
        print(ZombRand(num))
    end
end 
```
为了控制本地命名空间的污染，可以使用 do end 代码块，进一步改进：
```lua
local MyTable = {}
local ipairs = ipairs
local print = print
do
    -- 将非系统全局变量放入代码块中，避免本地命名空间污染
    local ZombRand = ZombRand
    function MyTable.Fun(table_list)
        for _, num in ipairs(table_list)
            print(ZombRand(num))
        end
    end
end
```

### 复合条件的顺序
考虑以下代码：
```lua
if x and y then
    -- do something
end
```
由于and和or逻辑判断是短路的，所以复合条件的顺序一定程度上能够控制逻辑判断次数。

如果实现知道条件的真假概率，显而易见：
>对于and逻辑，应该把结果为false的可能性较大的条件放在前面。\
对于or逻辑，应该把结果为true的可能性较大的条件放在前面。

### 避免重复调用
```lua
if getPlayer():getInventory():contains("MyMod.MyItem") and getPlayer():getInventory():contains("MyMod.MyItem") then
```
如果要多次使用同一个函数的返回值，确定其不变的情况下，将结果缓存在局部变量中：
```lua
local inventory = getPlayer():getInventory()
if inventory:contains("MyMod.MyItem") and inventory:contains("MyMod.MyItem") then
```

## 参考资料
https://github.com/FWolfe/Zomboid-Modding-Guide/blob/master/api/README.md