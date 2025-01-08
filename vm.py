"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.12/Include/internal/pycore_frame.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None

        self.instruction_index = 0
        self.instructions = list(dis.get_instructions(self.code))
        self.jump_targets = {inst.offset: i for i, inst in enumerate(self.instructions) if inst.is_jump_target}
        self.is_generator = False

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        if self.data_stack:
            return self.data_stack.pop()
        return None

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        while self.instruction_index < len(self.instructions):
            instruction = self.instructions[self.instruction_index]
            getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
            self.instruction_index += 1
        return self.return_value

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        pass

    def copy_op(self, arg: tp.Any) -> None:
        self.push(self.top())

    def jump(self, arg: int) -> None:
        self.instruction_index = self.jump_targets[arg] - 1

    def jump_forward_op(self, arg: int) -> None:
        self.jump(arg)

    def jump_backward_op(self, arg: int) -> None:
        self.jump(arg)

    def jump_if_true_or_pop_op(self, arg: int) -> None:
        self.jump(arg) if self.top() else self.pop()

    def jump_if_false_or_pop_op(self, arg: int) -> None:
        self.jump(arg) if not self.top() else self.pop()

    def pop_jump_if_true_op(self, arg: int) -> None:
        self.jump(arg) if self.pop() else None

    def pop_jump_if_false_op(self, arg: int) -> None:
        self.jump(arg) if not self.pop() else None

    def pop_jump_if_none_op(self, arg: int) -> None:
        if self.pop() is None:
            self.jump(arg)
        else:
            self.instruction_index += 1

    def pop_top_op(self, arg: tp.Any) -> None:
        if self.is_generator and self.top() is not None:
            return
        self.pop()

    def nop_op(self, arg: tp.Any) -> None:
        pass

    def call_op(self, arg: int) -> None:
        args = self.popn(arg)
        obj = self.pop()
        if callable(obj):
            self.push(obj(*args))
        else:
            self.push(obj)

    def call_function_ex_op(self, arg: int) -> None:
        vars_ = self.pop() if arg else {}
        args_ = self.pop()
        func = self.pop()
        self.push(func(*args_, **vars_))

    def intrinsic_print(self, arg: tp.Any) -> None:
        print(self.pop())
        self.return_value = arg

    def intrinsic_import_star(self, arg: tp.Any) -> None:
        module_name = self.pop()
        self.push(__import__(module_name.__name__, self.globals, self.locals))
        for name in dir(module_name):
            if name not in self.locals:
                self.locals[name] = getattr(module_name, name)
            if name not in self.globals:
                self.globals[name] = getattr(module_name, name)

    def intrinsic_list_to_tuple(self, arg: tp.Any) -> None:
        list_ = self.pop()
        result = tuple(list_)
        self.push(result)

    def call_intrinsic_1_op(self, arg: tp.Any) -> None:
        intrinsic_map = {
            "INTRINSIC_PRINT": self.intrinsic_print,
            "INTRINSIC_IMPORT_STAR": self.intrinsic_import_star,
            "INTRINSIC_LIST_TO_TUPLE": self.intrinsic_list_to_tuple,
        }
        operand = self.instructions[self.instruction_index].argrepr
        if operand in intrinsic_map:
            intrinsic_map[operand](arg)

    def load_fast_op(self, arg: str) -> None:
        self.push(self.locals[arg]) if arg in self.locals else None

    def load_name_op(self, arg: str) -> None:
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_fast_and_clear_op(self, arg: str) -> None:
        value = self.locals[arg] if arg in self.locals else None
        self.locals[arg] = None
        self.push(value)

    def load_global_op(self, arg: str) -> None:
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_locals_op(self, arg: str) -> None:
        self.push(self.locals)

    def load_const_op(self, arg: tp.Any) -> None:
        self.push(arg)

    def load_attr_op(self, arg: str) -> None:
        self.push(getattr(self.pop(), arg))

    def load_build_class_op(self, arg: None) -> None:
        self.push(builtins.__build_class__)

    def setup_annotations_op(self, arg: tp.Any) -> None:
        if '__annotations__' not in self.locals:
            self.locals['__annotations__'] = {}

    def store_slice_op(self, arg: tp.Any) -> None:
        value, seq, start, stop = self.popn(4)
        seq[start:stop] = value
        self.push(seq)

    def store_name_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    def store_global_op(self, arg: str) -> None:
        self.globals[arg] = self.pop()

    def store_fast_op(self, arg: str) -> None:
        self.store_name_op(arg)

    def store_attr_op(self, arg: str) -> None:
        value, object_ = self.popn(2)
        setattr(object_, arg, value)

    def store_subscr_op(self, arg: tp.Any) -> None:
        value, seq, key = self.popn(3)
        seq[key] = value

    def return_value_op(self, arg: tp.Any) -> None:
        self.return_value = self.pop()
        self.instruction_index = len(list(dis.get_instructions(self.code)))

    def return_const_op(self, arg: tp.Any) -> None:
        if not self.is_generator:
            self.return_value = arg
            self.instruction_index = len(list(dis.get_instructions(self.code)))

    def make_function_op(self, arg: int) -> None:
        code = self.pop()

        for flag in [0x08, 0x04]:
            self.pop() if arg & flag else None

        kw_defaults = self.pop() if arg & 0x02 else {}
        defaults = self.pop() if arg & 0x01 else {}

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            function_locals = {}
            has_args = bool(code.co_flags & 4)
            has_kwargs = bool(code.co_flags & 8)

            posonly_names = code.co_varnames[:code.co_posonlyargcount]
            arg_names = code.co_varnames[code.co_posonlyargcount:code.co_argcount]
            kwonly_names = code.co_varnames[code.co_argcount:]

            if has_args:
                function_locals[kwonly_names[code.co_kwonlyargcount]] = tuple(args[code.co_argcount:])
            if has_kwargs:
                function_locals[kwonly_names[code.co_kwonlyargcount + has_args]] = {}  # type: ignore

            for name, value in zip(posonly_names + arg_names, args):
                function_locals[name] = value
            for name, value in kwargs.items():
                function_locals[kwonly_names[code.co_kwonlyargcount + has_args]][name] = value  # type: ignore

            if defaults:
                pos_names = code.co_varnames[:code.co_argcount]
                for name, default_value in zip(pos_names[code.co_argcount - len(defaults):], defaults):
                    if name not in function_locals:
                        function_locals[name] = default_value
            if kw_defaults:
                for name, default_value in kw_defaults.items():
                    if name not in function_locals:
                        function_locals[name] = default_value

            return Frame(code, self.builtins, self.globals, {**self.locals, **function_locals}).run()

        self.push(f)

    def is_op_op(self, arg: int) -> None:
        b = self.pop()
        a = self.pop()
        if arg:
            self.push(a is not b)
        else:
            self.push(a is b)

    def contains_op_op(self, arg: int) -> None:
        b = self.pop()
        a = self.pop()
        if arg:
            self.push(a not in b)
        else:
            self.push(a in b)

    def unary_positive_op(self, arg: tp.Any) -> None:
        item = self.pop()
        self.push(item)

    def unary_negative_op(self, arg: tp.Any) -> None:
        item = self.pop()
        self.push(-item)

    def unary_not_op(self, arg: tp.Any) -> None:
        item = self.pop()
        self.push(not item)

    def unary_invert_op(self, arg: tp.Any) -> None:
        item = self.pop()
        self.push(~item)

    def binary_slice_op(self, arg: tp.Any) -> None:
        end = self.pop()
        start = self.pop()
        seq = self.pop()
        self.push(seq[start:end])

    def binary_op_op(self, arg: tp.Any) -> None:
        b = self.pop()
        a = self.pop()

        binary_op_map: dict[str, tp.Callable[[tp.Any, tp.Any], tp.Any]] = {
            '+': lambda lhs, rhs: lhs + rhs,
            '+=': lambda lhs, rhs: lhs + rhs,
            '-': lambda lhs, rhs: lhs - rhs,
            '-=': lambda lhs, rhs: lhs - rhs,
            '*': lambda lhs, rhs: lhs * rhs,
            '*=': lambda lhs, rhs: lhs * rhs,
            '%': lambda lhs, rhs: lhs % rhs,
            '%=': lambda lhs, rhs: lhs % rhs,
            '**': lambda lhs, rhs: lhs ** rhs,
            '**=': lambda lhs, rhs: lhs ** rhs,
            '/': lambda lhs, rhs: lhs / rhs,
            '/=': lambda lhs, rhs: lhs / rhs,
            '//': lambda lhs, rhs: lhs // rhs,
            '//=': lambda lhs, rhs: lhs // rhs,
            '<<': lambda lhs, rhs: lhs << rhs,
            '<<=': lambda lhs, rhs: lhs << rhs,
            '>>': lambda lhs, rhs: lhs >> rhs,
            '>>=': lambda lhs, rhs: lhs >> rhs,
            '&': lambda lhs, rhs: lhs & rhs,
            '&=': lambda lhs, rhs: lhs & rhs,
            '|': lambda lhs, rhs: lhs | rhs,
            '|=': lambda lhs, rhs: lhs | rhs,
            '^': lambda lhs, rhs: lhs ^ rhs,
            '^=': lambda lhs, rhs: lhs ^ rhs,
        }

        operation = self.instructions[self.instruction_index].argrepr
        if operation in binary_op_map:
            self.push(binary_op_map[operation](a, b))

    def binary(self, arg: str) -> None:
        b: tp.Any = self.pop()
        a: tp.Any = self.pop()
        binary_map: dict[str, tp.Callable[[tp.Any, tp.Any], tp.Any]] = {
            '+': lambda lhs, rhs: lhs + rhs,
            '*': lambda lhs, rhs: lhs * rhs,
            '-': lambda lhs, rhs: lhs - rhs,
            '/': lambda lhs, rhs: lhs / rhs,
            '//': lambda lhs, rhs: lhs // rhs,
            'and': lambda lhs, rhs: lhs & rhs,
            'or': lambda lhs, rhs: lhs | rhs,
            '**': lambda lhs, rhs: lhs ** rhs,
            '@': lambda lhs, rhs: lhs @ rhs,
            '%': lambda lhs, rhs: lhs % rhs,
            '[]': lambda lhs, rhs: lhs[rhs],
            '<<': lambda lhs, rhs: lhs << rhs,
            '>>': lambda lhs, rhs: lhs >> rhs,
            '^': lambda lhs, rhs: lhs ^ rhs
        }
        if arg in binary_map:
            self.push(binary_map[arg](a, b))

    def binary_add_op(self, arg: tp.Any) -> None:
        self.binary('+')

    def binary_multiply_op(self, arg: tp.Any) -> None:
        self.binary('*')

    def binary_subtract_op(self, arg: tp.Any) -> None:
        self.binary('-')

    def binary_true_divide_op(self, arg: tp.Any) -> None:
        self.binary('/')

    def binary_floor_divide_op(self, arg: tp.Any) -> None:
        self.binary('//')

    def binary_and_op(self, arg: tp.Any) -> None:
        self.binary('and')

    def binary_or_op(self, arg: tp.Any) -> None:
        self.binary('or')

    def binary_power_op(self, arg: tp.Any) -> None:
        self.binary('**')

    def binary_matrix_multiply_op(self, arg: tp.Any) -> None:
        self.binary('@')

    def binary_modulo_op(self, arg: tp.Any) -> None:
        self.binary('%')

    def binary_lshift_op(self, arg: tp.Any) -> None:
        self.binary('<<')

    def binary_rshift_op(self, arg: tp.Any) -> None:
        self.binary('>>')

    def binary_subscr_op(self, arg: tp.Any) -> None:
        self.binary('[]')

    def binary_xor_op(self, arg: tp.Any) -> None:
        self.binary('^')

    def compare_op_op(self, arg: tp.Any) -> None:
        b = self.pop()
        a = self.pop()
        compare_map: dict[str, tp.Callable[[tp.Any, tp.Any], tp.Any]] = {
            '<': lambda lhs, rhs: lhs < rhs,
            '<=': lambda lhs, rhs: lhs <= rhs,
            '>': lambda lhs, rhs: lhs > rhs,
            '>=': lambda lhs, rhs: lhs >= rhs,
            '==': lambda lhs, rhs: lhs == rhs,
            '!=': lambda lhs, rhs: lhs != rhs,
        }
        self.push(compare_map[arg](a, b))

    def unpack_sequence_op(self, arg: int) -> None:
        sequence = self.pop()
        for item in reversed(sequence):
            self.push(item)

    def list_extend_op(self, arg: tp.Any) -> None:
        list2 = self.pop()
        list1 = self.pop()
        list1.extend(list2)
        self.push(list1)

    def get_iter_op(self, arg: tp.Any) -> None:
        self.push(iter(self.pop()))

    def for_iter_op(self, arg: tp.Any) -> None:
        try:
            iter_ = self.pop()
            value = next(iter_)
            self.push(iter_)
            self.push(value)
        except StopIteration:
            self.jump(arg)

    def end_for_op(self, arg: tp.Any) -> None:
        self.return_value = self.pop()

    def build_tuple_op(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(tuple(seq))

    def build_list_op(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(list(seq))

    def build_set_op(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(set(seq))

    def build_map_op(self, arg: int) -> None:
        seq = self.popn(2 * arg)
        map_ = dict()
        for i in range(0, 2 * arg, 2):
            map_[seq[i]] = seq[i - 1]
        self.push(map_)

    def build_const_key_map_op(self, arg: int) -> None:
        keys = self.pop()
        values = self.popn(arg)
        self.push(dict(zip(keys, values)))

    def build_slice_op(self, arg: int) -> None:
        seq = self.popn(arg)
        self.push(slice(*seq))

    def build_string_op(self, arg: int) -> None:
        values = map(str, self.popn(arg))
        self.push("".join(values))

    def delete_subscr_op(self, arg: tp.Any) -> None:
        seq, key = self.popn(2)
        del seq[key]

    def delete_name_op(self, arg: str) -> None:
        del self.locals[arg]

    def delete_global_op(self, arg: str) -> None:
        del self.globals[arg]

    def delete_fast_op(self, arg: str) -> None:
        del self.locals[arg]

    def delete_attr_op(self, arg: str) -> None:
        delattr(self.pop(), arg)

    def import_from_op(self, arg: str) -> None:
        module = self.top()
        self.push(getattr(module, arg))

    def import_name_op(self, arg: str) -> None:
        level, fromlist = self.popn(2)
        self.push(__import__(arg, self.globals, self.locals, fromlist, level))

    def raise_varargs_op(self, arg: int) -> None:
        if arg == 0:
            raise
        elif arg == 1:
            exc = self.pop()
            raise exc
        elif arg == 2:
            mes = self.pop()
            exc = self.pop()
            raise exc(mes)

    def swap_op(self, arg: tp.Any) -> None:
        self.data_stack[-1], self.data_stack[-arg] = self.data_stack[-arg], self.data_stack[-1]

    def dict_merge_op(self, arg: tp.Any) -> None:
        dict1, dict2 = self.popn(2)
        for key, value in dict2.items():
            dict1[key] = value
        self.push(dict1)

    def dict_update_op(self, arg: tp.Any) -> None:
        other = self.pop()
        original = self.pop()
        original.update(other)
        self.push(original)

    def set_update_op(self, arg: tp.Any) -> None:
        new_elements = self.pop()
        original_set = self.pop()
        original_set.update(new_elements)
        self.push(original_set)

    def format_value_op(self, arg: tp.Any) -> None:
        arg = self.instructions[self.instruction_index].arg
        if arg is None:
            return None
        specifier = self.pop() if arg & 4 else ""
        obj = self.pop()
        if arg % 4 == 0:
            self.push(f"{obj:{specifier}}")
        elif arg % 4 == 1:
            self.push(f"{obj!s:{specifier}}")
        elif arg % 4 == 2:
            self.push(f"{obj!r:{specifier}}")
        elif arg % 4 == 3:
            self.push(f"{obj!a:{specifier}}")


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
