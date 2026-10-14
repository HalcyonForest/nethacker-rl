from pathlib import Path
from runtime.numpy_actor import NumpyActor

def make_agent():
    return NumpyActor(Path(__file__).with_name('actor.npz'))
