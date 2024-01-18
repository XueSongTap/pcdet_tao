# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PointPillars decorators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

"""
这段代码定义了两个Python装饰器，override和subclass，它们用于增强Python类继承时方法重写的清晰性和验证。
"""
def override(method):
    """
    Override decorator.

    Decorator implementing method overriding in python
    Must also use the @subclass class decorator
    
    override 装饰器用于明确表明一个方法是用来覆盖基类中的同名方法。这样做有助于代码的阅读者理解该方法的用途，并且可以通过subclass装饰器来进行验证，确保真的有一个同名的方法可以被覆盖。

装饰器的使用方式是在你定义的方法前面加上@override。当你这么做时，装饰器会在方法对象上设置一个名为override的属性，并将其值设为True，然后返回这个方法。这个装饰器并不改变方法的行为，它只是用于标记和以后的验证。
    """
    
    method.override = True
    return method


def subclass(class_object):
    """
    Subclass decorator.

    Verify all @override methods
    Use a class decorator to find the method's class
    subclass 装饰器应用在类定义上。它会检查类中所有标记为@override的方法是否确实覆盖了基类中的方法。如果一个标记为@override的方法在任何基类中没有找到，subclass装饰器将断言失败，并抛出一个错误，指出未在任何基类中找到该方法。

    此外，如果一个标记为@override的方法自身没有文档字符串（docstring），则subclass装饰器会从基类的同名方法中复制docstring，这样做可以保持文档的一致性。

    装饰器的工作原理是通过class_object.__dict__.items()来遍历类定义中的所有属性和方法，如果一个方法有override属性，则进一步通过inspect.getmro(class_object)获取类的继承关系。然后它会检查这个方法是否出现在除了自身类以外的基类的字典中。如果找到了，确认这个方法是有效的重写，否则抛出断言错误。
    """
    for name, method in class_object.__dict__.items():
        if hasattr(method, "override"):
            found = False
            for base_class in inspect.getmro(class_object)[1:]:
                if name in base_class.__dict__:
                    if not method.__doc__:
                        # copy docstring
                        method.__doc__ = base_class.__dict__[name].__doc__
                    found = True
                    break
            assert found, '"%s.%s" not found in any base class' % (
                class_object.__name__,
                name,
            )
    return class_object
