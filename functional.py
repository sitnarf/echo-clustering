from copy import copy
from functools import reduce
from typing import Callable, TypeVar, Iterable, Tuple, Dict, List, Any, Union

from toolz import curry

from custom_types import IndexAccess

T1 = TypeVar('T1')
T2 = TypeVar('T2')


def t(*args):
    print(*args)
    return args[-1]


def or_fn(*fns: Callable[..., bool]) -> Callable[..., bool]:
    return lambda *args: reduce(lambda current_value, fn: current_value or fn(*args), fns, False)


def decorate_unpack(function: Callable[..., T1]) -> Callable[[Iterable], T1]:

    def unpacked(args):
        return function(*args)

    return unpacked


def dict_from_items(items: Iterable[Tuple[T1, T2]]) -> Dict[T1, T2]:
    out_dict = {}
    for key, value in items:
        out_dict[key] = value
    return out_dict


def list_from_items(items: Iterable[Tuple[int, T1]]) -> List[T1]:
    out_list: List[T1] = []
    for key, value in items:
        out_list.insert(key, value)
    return out_list


def mapl(func, iterable):
    return list(map(func, iterable))


def mapi(func, iterable):
    return map(decorate_unpack(func), enumerate(iterable))


def map_tuples(callback: Callable[..., T2], iterable: Iterable[T1]) -> Iterable[T2]:
    return map(decorate_unpack(callback), iterable)  # type: ignore


def dict_subset(list_keys: List[str], dictionary: dict) -> dict:
    return {k: dictionary[k] for k in list_keys}


def dict_subset_list(list_keys: List[str], dictionary: dict) -> List:
    return [dictionary[k] for k in list_keys]


def flatten(iterable_outer: Iterable[Union[Iterable[T1], T1]]) -> Iterable[T1]:
    for iterable_inner in iterable_outer:
        if isinstance(iterable_inner, Iterable) and not isinstance(iterable_inner, str):
            for item in iterable_inner:
                yield item
        else:
            yield iterable_inner  # type: ignore


def find(callback: Callable[[T1], bool], list_to_search: Iterable[T1]) -> T1:
    return next(filter(callback, list_to_search))


def find_index(
    callback: Callable[[T1], bool], list_to_search: Union[List[T1], str], reverse=False
) -> int:
    if reverse:
        iterable = add_index_reversed(list_to_search)
    else:
        iterable = add_index(list_to_search)
    return next(filter(lambda item: callback(item[1]), iterable))[0]


def add_index(iterable: Iterable) -> Iterable:
    for index, item in enumerate(iterable):
        yield index, item


def add_index_reversed(iterable: Union[List, str]) -> Iterable:
    for index in reversed(range(len(iterable))):
        yield index, iterable[index]


def do_nothing():
    pass


def pass_value() -> Callable[[T1], T1]:

    def pass_value_callback(value):
        return value

    return pass_value_callback


def in_ci(string: str, sequence: Union[List, str]) -> bool:
    normalized_sequence = [item.upper()
                           for item in sequence] if isinstance(sequence, List) else sequence.upper()
    return string.upper() in normalized_sequence


def partial_method(method: Callable, *args, **kwargs) -> Callable:

    def partial_callback(self: object):
        method(self, *args, **kwargs)

    return partial_callback


def pipe(*args: Any) -> Any:
    current_value = args[0]
    for function in args[1:]:
        current_value = function(current_value)
    return current_value


def piped(*args: Any) -> Any:
    current_value = args[0]
    for function in args[1:]:
        breakpoint()
        current_value = function(current_value)
    return current_value


def lambda_with_consts(define, to):
    return lambda: to(*define)


def pass_args(define, to):
    return to(*define)


def define_consts(to: Callable[..., T1], **variables) -> T1:
    return to(**variables)


def statements(*args: Any) -> Any:
    return args[-1]


def filter_keys(keys: Iterable[Any], dictionary: Dict) -> Dict:
    return {key: dictionary[key] for key in keys}


def unzip(iterable: Iterable) -> Iterable:
    return zip(*iterable)


def try_except(try_clause: Callable, except_clauses: Dict) -> Any:
    # noinspection PyBroadException
    try:
        return try_clause()
    # noinspection PyBroadException
    except Exception as e:
        for ExceptionClass, except_clause in except_clauses.items():
            if isinstance(e, ExceptionClass):
                return except_clause()
        raise e


def merge_by(callback: Callable, sequence: Iterable) -> Any:
    sequence_iterable = iter(sequence)
    last_item = next(sequence_iterable)
    for item in sequence_iterable:
        last_item = callback(last_item, item)
    return last_item


TIndexAccess = TypeVar('TIndexAccess', bound=IndexAccess)


def assign_index(something: TIndexAccess, index: Any, value: Any) -> TIndexAccess:
    something_copied = copy(something)
    something_copied[index] = value
    return something_copied


def skip_first(input_iterable: Iterable) -> Iterable:
    is_first = True
    for item in input_iterable:
        if is_first:
            is_first = False
            continue
        yield item


def iter_is_last(input_iterable: Iterable) -> Iterable:
    iterator = iter(input_iterable)
    last_value = next(iterator)
    for value in iterator:
        yield last_value, False
        last_value = value
    yield last_value, True


@curry
def find_index_right(function: Callable, list_: List) -> Any:
    for index, value in reversed(list(enumerate(list_))):
        if function(index, value):
            return index, value


def does_objects_equal(obj1: Any, obj2: Any) -> bool:
    try:
        obj1dict = obj1.__dict__
        obj2dict = obj2.__dict__
    except AttributeError:
        return obj1 == obj2
    else:
        return obj2dict == obj1dict


def raise_exception(exception: Any) -> None:
    raise exception


def compact(iterable: Iterable) -> Iterable:
    return filter(lambda i: i is not None, iterable)
