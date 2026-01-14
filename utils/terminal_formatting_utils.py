######################################################
#                                                    #
#                      MiniGrid-LaC                  #
#                        PROGRAM                     #
#                                                    #
######################################################


"""
Colored and formatted terminal printing utilities.

This module provides a helper function `cprint` that allows printing
text to the terminal with colors and text styles such as:
- bold
- dim
- underline
- italic
- reverse video

It is useful for debugging, logging, or improving readability
in CLI-based applications (e.g., RL environments, VLM pipelines).
"""




######################################################
#                                                    #
#                      LIBRARIES                     #
#                                                    #
######################################################


# ------------------------------------------------------------
# colorama: Cross-platform colored terminal text
# ------------------------------------------------------------
# colorama ensures ANSI escape codes work correctly on Windows,
# while remaining compatible with Linux/macOS terminals.
from colorama import Fore, Style, init




######################################################
#                                                    #
#                   INITIALIZATION                   #
#                         &                          #
#                  GLOBAL VARIABLES                  #
#                                                    #
######################################################


# Initialize colorama.
# autoreset=True ensures styles do not "leak" to subsequent prints.
init(autoreset=True)

# ------------------------------------------------------------
# ANSI escape codes (not directly provided by colorama)
# ------------------------------------------------------------
# These are standard terminal control sequences supported by
# most modern terminals (Linux, macOS, VSCode terminal).
ANSI_UNDERLINE = "\033[4m"   # Underlined text
ANSI_ITALIC = "\033[3m"      # Italic text (not supported everywhere)
ANSI_REVERSE = "\033[7m"     # Reverse foreground/background colors

# Colors that can be used
WHITE = Fore.WHITE
LIGHT_WHITE = Fore.LIGHTWHITE_EX
BLACK = Fore.BLACK
LIGHT_BLACK = Fore.LIGHTBLACK_EX
BLUE = Fore.BLUE
LIGHT_BLUE = Fore.LIGHTBLUE_EX
RED = Fore.RED
LIGHT_RED = Fore.LIGHTRED_EX
GREEN = Fore.GREEN
LIGHT_GREEN = Fore.LIGHTGREEN_EX
YELLOW = Fore.YELLOW
LIGHT_YELLOW = Fore.LIGHTYELLOW_EX
MAGENTA = Fore.MAGENTA
LIGHT_MAGENTA = Fore.LIGHTMAGENTA_EX
CYAN = Fore.CYAN
LIGHT_CYAN = Fore.LIGHTCYAN_EX




######################################################
#                                                    #
#                      FUNCTION                      #
#                                                    #
######################################################


def cprint(text: str,
           color=Fore.WHITE,
           bold: bool = False,
           dim: bool = False,
           underline: bool = False,
           italic: bool = False,
           reverse: bool = False,
           indent: int = 0,
           ):
    """
    Print formatted and colored text to the terminal.

    Parameters
    ----------
    text : str
        The text to display in the terminal.

    color : colorama.Fore (default: Fore.WHITE)
        Foreground color of the text (e.g., Fore.RED, Fore.GREEN).

    bold : bool
        If True, display text in bold/bright style.

    dim : bool
        If True, display text in dim (low-intensity) style.

    underline : bool
        If True, underline the text (ANSI feature).

    italic : bool
        If True, display italic text (ANSI feature, terminal-dependent).

    reverse : bool
        If True, reverse foreground and background colors.
    
    indent : int
        The amount of indentation we want for our text.

    Notes
    -----
    - Styles are combined dynamically.
    - Unsupported styles are silently ignored by the terminal.
    """

    # Accumulate all requested styles into a single prefix string
    style = ""

    if bold:
        style += Style.BRIGHT     # Bright/Bold text

    if dim:
        style += Style.DIM        # Dim text

    if underline:
        style += ANSI_UNDERLINE   # Underlined text

    if italic:
        style += ANSI_ITALIC      # Italic text

    if reverse:
        style += ANSI_REVERSE     # Reverse video
    
    indent_str = " " * indent

    # Print the final formatted message
    # Style.RESET_ALL ensures formatting stops after this print
    print(f"{indent_str}{style}{color}{text}{Style.RESET_ALL}")




######################################################
#                                                    #
#                       TESTING                      #
#                                                    #
######################################################


# ------------------------------------------------------------
# Test / demo code (runs only when executed directly)
# ------------------------------------------------------------
def _demo():
    """Demonstrate available styles and colors."""
    cprint("INFO message", Fore.CYAN)
    cprint("SUCCESS message", Fore.GREEN, bold=True)
    cprint("WARNING message", Fore.YELLOW, bold=True)
    cprint("ERROR message", Fore.RED, bold=True, underline=True)
    cprint("HIGHLIGHT", Fore.WHITE, reverse=True)
    cprint("Italic text", Fore.MAGENTA, italic=True)
    cprint("Dim text", Fore.WHITE, dim=True)




######################################################
#                                                    #
#                        MAIN                        #
#                                                    #
######################################################


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        print("Running tests")
        _demo()