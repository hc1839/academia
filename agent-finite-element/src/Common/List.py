r"""
List operations inspired by functional programming.
"""

class AssocList(object):
    r"""
    Association list as a list of pairs. Order of the pairs is maintained.

    **Data members:**
        ``_eqFn``: See :func:`__init__`.

        ``_assocList``: List of pairs, where first component of a pair is key,
        and second component is associated value.
    """

    def __init__(self, eqFn, sequence = None):
        r"""
        :param callable eqFn:
            Binary function that returns ``True`` if two keys are considered
            equal; ``False`` if otherwise.

        :param iterable sequence:
            Sequence of pairs to initialize the association list.

        :exception ValueError:
            Elements of the sequence are not pairs.
        """
        if (not(callable(eqFn))):
            raise TypeError()

        self._eqFn = eqFn

        if (sequence is None):
            self._assocList = []
        else:
            sequence = list(sequence)

            for elt in sequence:
                if (not(isinstance(elt, tuple) and len(elt) == 2)):
                    raise TypeError('Elements of the sequence are not pairs.')

            self._assocList = sequence

    def __iter__(self):
        r"""
        Iterator of the pairs in the association list.
        """
        for pair in self._assocList:
            yield pair

    def __getitem__(self, key):
        r"""
        :param key:
            Key associated with the value to get.

        :return:
            Value associated with the first occurrence of ``key``.

        :exception KeyError:
            Key does not exist.
        """
        for k, v in self:
            if (self._eqFn(k, key)):
                return v

        raise KeyError('Key does not exist.')

    def __len__(self):
        r"""
        Number of pairs in the association list.
        """
        return len(self._assocList)

    def add(self, key, value):
        r"""
        Adds a key-value association to the beginning of the list.

        :param key:
            Key of the pair.

        :param value:
            Value of the pair.

        :return:
            ``None``
        """
        self._assocList = [(key, value)] + self._assocList

    def remove(self, key):
        r"""
        Removes the first pair with a given key.

        :param key:
            Key of the first pair to be removed.

        :exception KeyError:
            Key does not exist.

        :return:
            ``None``
        """
        assocList = self._assocList

        for i in range(len(assocList)):
            if (self._eqFn(assocList[i][0], key)):
                del self._assocList[i:i+1]
                return

        raise KeyError('Key does not exist.')

    def isMember(self, key):
        r"""
        Determines whether a given key exists.

        :param key:
            Key to test for existence.

        :return:
            ``True`` if ``key`` exists as a key; ``False`` if otherwise.

        :rtype: bool
        """
        try:
            self[key]
        except KeyError:
            return False

        return True
