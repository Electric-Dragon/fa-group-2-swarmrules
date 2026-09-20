from fa2026 import execute, load_scenario
from .playground_controller import act, setup


scenario = load_scenario("playground")
result = execute(
    scenario,
    setup,
    act,
    visualize=True,
    delay=0.02,
    draw_every=2,
    max_turns=1000,
)
print(result)
