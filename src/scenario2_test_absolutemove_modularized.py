"""
Scenario 2 experiment environment test script (absolute movement version - modularized)

Scenario 2: Go to the blue pillar, turn right, then stop next to the table.

This script uses the modularized ScenarioExperiment class from utils.miscellaneous.scenario_runner,
similar to minigrid_lac.py structure.

> Maps are now loaded from JSON files. To change the map, simply modify the JSON file.

Usage:
    # Run from src/ directory
    cd src/
    python scenario2_test_absolutemove_modularized.py [json_map_path]
    
    Examples:
    python scenario2_test_absolutemove_modularized.py config/example_map.json
    python scenario2_test_absolutemove_modularized.py config/scenario135_example_map.json
    
    # Show help
    python scenario2_test_absolutemove_modularized.py --help
"""

import sys

# Import modularized components
import utils.prompt_manager.terminal_formatting_utils as tfu
from utils.miscellaneous.scenario_runner import ScenarioExperiment
from utils.miscellaneous.safe_minigrid_registration import safe_minigrid_reg
from utils.miscellaneous.global_variables import MAP_FILE_NAME, LOGPROBS_ENABLED, DEBUG

# MiniGrid Environment Safe Registration
safe_minigrid_reg()


def main():
    """Main function for Scenario 2 experiment (modularized version)"""
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            tfu.cprint("-- Usage Instructions --", tfu.LIGHT_GREEN, bold=True)
            tfu.cprint("python scenario2_test_absolutemove_modularized.py [json_map_path]", tfu.LIGHT_RED, italic=True, indent=8)
            tfu.cprint(f"Example: python scenario2_test_absolutemove_modularized.py config/{MAP_FILE_NAME}", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint(f"Default: Uses MAP_FILE_NAME from global_variables.py (currently: {MAP_FILE_NAME})", tfu.LIGHT_BLACK, italic=True)
            tfu.cprint("\nNote: Run from src/ directory", tfu.LIGHT_BLACK, italic=True)
            return
        else:
            json_map_path = sys.argv[1]
    else:
        # Use global MAP_FILE_NAME (will be set to config/{MAP_FILE_NAME} in ScenarioExperiment.__init__)
        json_map_path = None
    
    try:
        # Create and run experiment using modularized ScenarioExperiment
        # use_logprobs, debug: from global_variables.py
        # System prompt: USE_VERBALIZED_ENTROPY (global_variables.py) — False → system_prompt_start.txt, True → system_prompt_verbalized_entropy.txt
        experiment = ScenarioExperiment(
            json_map_path=json_map_path,
            use_logprobs=LOGPROBS_ENABLED,
            debug=DEBUG,
        )
        experiment.run()
        experiment.cleanup()
    except KeyboardInterrupt:
        tfu.cprint("\n\nInterrupted by user.", tfu.LIGHT_BLUE, bold=True)
        if 'experiment' in locals():
            experiment.cleanup()
    except Exception as e:
        tfu.cprint(f"\n\nError occurred: {e}", tfu.LIGHT_RED, bold=True)
        import traceback
        traceback.print_exc()
        if 'experiment' in locals():
            experiment.cleanup()


if __name__ == "__main__":
    main()

