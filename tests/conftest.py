import sys
import types

# Dummy client classes used to mock external dependencies
class DummyClient:
    def __init__(self, *args, **kwargs):
        pass
    async def create(self, *args, **kwargs):
        return {}

# Minimal numpy replacement used by validate_predictions
class DummyArray(list):
    def any(self):
        return any(self)
    def all(self):
        return all(self)

class DummyNumpy(types.ModuleType):
    def array(self, x):
        return DummyArray(x)
    def isnan(self, arr):
        return DummyArray(False for _ in arr)
    def isinf(self, arr):
        return DummyArray(False for _ in arr)
    def max(self, arr):
        return max(arr)
    def abs(self, arr):
        return DummyArray(abs(v) for v in arr)
    def unique(self, arr):
        seen = set()
        uniq = []
        for v in arr:
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq
    def isfinite(self, value):
        return not (value != value or value in (float('inf'), float('-inf')))

# Simple mean squared error function
def dummy_mean_squared_error(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)

modules = {
    "autogen_agentchat": types.ModuleType("autogen_agentchat"),
    "autogen_agentchat.agents": types.ModuleType("autogen_agentchat.agents"),
    "autogen_agentchat.teams": types.ModuleType("autogen_agentchat.teams"),
    "autogen_agentchat.ui": types.ModuleType("autogen_agentchat.ui"),
    "autogen_agentchat.conditions": types.ModuleType("autogen_agentchat.conditions"),
    "autogen_core": types.ModuleType("autogen_core"),
    "autogen_core.tools": types.ModuleType("autogen_core.tools"),
    "autogen_ext": types.ModuleType("autogen_ext"),
    "autogen_ext.models": types.ModuleType("autogen_ext.models"),
    "autogen_ext.models.openai": types.ModuleType("autogen_ext.models.openai"),
    "autogen_ext.models.anthropic": types.ModuleType("autogen_ext.models.anthropic"),
    "openai": types.ModuleType("openai"),
    "numpy": DummyNumpy("numpy"),
    "pandas": types.ModuleType("pandas"),
    "matplotlib": types.ModuleType("matplotlib"),
    "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
    "seaborn": types.ModuleType("seaborn"),
    "yfinance": types.ModuleType("yfinance"),
    "sklearn.metrics": types.ModuleType("sklearn.metrics"),
    "sklearn.model_selection": types.ModuleType("sklearn.model_selection"),
    "sklearn.ensemble": types.ModuleType("sklearn.ensemble"),
    "sklearn.linear_model": types.ModuleType("sklearn.linear_model"),
}

modules["autogen_agentchat.agents"].AssistantAgent = object
modules["autogen_agentchat.teams"].RoundRobinGroupChat = object
modules["autogen_agentchat.ui"].Console = object
modules["autogen_agentchat.conditions"].TextMentionTermination = object
modules["autogen_core.tools"].FunctionTool = object
modules["autogen_ext.models.openai"].OpenAIChatCompletionClient = DummyClient
modules["autogen_ext.models.anthropic"].AnthropicChatCompletionClient = DummyClient
modules["sklearn.metrics"].mean_squared_error = dummy_mean_squared_error
modules["sklearn.metrics"].accuracy_score = lambda a, b: 0.0
modules["sklearn.model_selection"].train_test_split = lambda *a, **k: (a[0], a[0], a[0], a[0])
class DummyRegressor:
    def fit(self, X, y):
        return self
    def predict(self, X):
        return [0] * len(X)
modules["sklearn.ensemble"].RandomForestRegressor = DummyRegressor
modules["sklearn.linear_model"].LinearRegression = DummyRegressor
class DummySeries(list):
    pass
modules["pandas"].Series = DummySeries
modules["pandas"].DataFrame = DummySeries

for name, module in modules.items():
    sys.modules.setdefault(name, module)
modules["matplotlib"].use = lambda *a, **k: None
