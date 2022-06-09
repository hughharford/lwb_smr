Feature: CodeSignal c002 century from year
    2nd challenge, using parameters

    Scenario Outline: Return the century from the year input
        Given a the "<inputyear>" as input
        When a calculation is made
        Then the expected 2 century is "<centuryexpected>"

        Examples:
            | inputyear | centuryexpected |
            | 2001      | 21              |
            | 98        | 1               |
            | 1656      | 17              |