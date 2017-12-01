r"""
Event-related functionalities.
"""

import abc

from List import AssocList

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class Event(ABC):
    r"""
    Abstract base class for events.

    **Data members:**
        ``_handlers``: :class:`List.AssocList` of (o, (h, a)) pairs, where
        ``o`` is observer, ``h`` is handler (a member function of ``o``), and
        ``a`` is list of any additional values.
    """

    def __init__(self):
        self._handlers = AssocList(lambda a, b: a is b, [])

    @abc.abstractmethod
    def notify(self):
        r"""
        ``[pure virtual]`` Notifies observers by calling their event handlers.

        :return:
            ``None``
        """
        pass

    def attach(self, observer, handler, *argv):
        r"""
        Attaches a function as an event handler.

        :param object observer:
            Observer of the event.

        :param callable handler:
            Function to be attached as a handler of the event. It is a member
            function of the class of ``observer`` and, therefore, must receive
            ``observer`` as its first argument.

        :param argv:
            Any additional values.

        :return:
            ``None``

        :exception ValueError:
            Observer is already listening to the event.
        """
        if (not(callable(handler))):
            raise TypeError()

        if (self._handlers.isMember(observer)):
            raise ValueError('Observer is already listening to the event.')

        self._handlers.add(observer, (handler, argv))

    def detach(self, observer):
        r"""
        Detaches an observer.

        :param observer:
            Observer to be detached from receiving the event.

        :return:
            ``None``

        :exception ValueError:
            Observer is not listening to the event.
        """
        if (not(self._handlers.isMember(observer))):
            raise ValueError('Observer is not listening to the event.')

        self._handlers.remove(observer)
