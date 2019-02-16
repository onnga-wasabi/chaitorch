import contextlib


class Reporter(object):

    def __init__(self):
        self._observer_names = {}
        self.observation = {}

    def __enter__(self):
        _reporters.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        _reporters.pop()

    @contextlib.contextmanager
    def scope(self, observation):
        old = self.observation
        self.observation = observation
        self.__enter__()
        yield
        self.__exit__(None, None, None)
        self.observation = old

    def add_observer(self, name, observer):
        self._observer_names[id(observer)] = name

    def add_observers(self, observers):
        for name, observer in observers:
            self._observer_names[id(observer)] = name

    def report(self, values, observer=None):
        if observer:
            observer_name = self._observer_names[id(observer)]
            for key, value in values.items():
                name = f'{observer_name}/{key}'
                self.observation[name] = value
        else:
            self.observation.update(values)


_reporters = []


def get_current_reporter():
    return _reporters[-1]


def report(values, observer=None):
    if _reporters:
        current = _reporters[-1]
        current.report(values, observer)


class Summarizer(object):

    def __init__(self, observations):
        self.observations = observations
        self.len = 0

    def add(self, observation):
        for key, value in observation.items():
            self.observations[key] += value
        self.len += 1

    def compute_mean(self):
        return {k: v / self.len for k, v in self.observations.items()}
