def exit_loop(prompt="Continue? [ENTER / q=quit]: "):
    """Pause execution during loops for debugging.

    Raises
    ------
    KeyboardInterrupt
        If the user enters 'q', to stop the loop/program gracefully.
    """
    ans = input(prompt)
    if ans.lower() == "q":
        raise KeyboardInterrupt("Loop interrupted by user.")