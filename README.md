# Piggy
To piggy-bank or roll? That's the decision you face when playing the the strategic dice game 'Pig'. The rules are simple but playing optimally is snout as easy as it may seem...

### The Rules

[Pig](https://en.wikipedia.org/wiki/Pig_(dice_game) is a strategic, two-player dice game. The rules are simple - each 
turn, a player repeatedly rolls a die until they either roll a 1 or they decide to 'hold':

- If a 1 is rolled, the player looses all accumulated points in that turn and it becomes the other player's turn
- If any other number is rolled, that number gets added to their turn total and the player decides whether to roll again or hold
- If a player chooses to hold, their turn total is added to their overall score and it becomes the other player's turn

The first player to reach 100 points wins!

### Basic Strategy Will Only Take You Sow Far

It is relatively straightforward to figure out that in order to maximise the expected number of points added to your score in
a single turn, you should continue to roll until your turn score reaches 20. Whilst this simple strategy will typically serve you well,
there are many circumstances in which you'd do best to deviate from it. The reason for this is that maximising the number of 
points you add to your score in a single turn is **not** the same thing as maximising your changes of winning. Consider, 
example, the extreme case in which your opponents sits on 99 points with you far behind at just 50. In this situation the
'hold at 20' policy is clearly no longer useful as your opponent is most certain to win on their next go.

### Finding the Optimal Policy

With a bit of thought it's easy to figure out _generally_ under what circumstances you should play with more or less risk.
And yet, determining _precisely_ the optimal strategy in each of the 100^3 possible states is clearly far beyond human
comprehension. Thankfully, given our complete understanding of the game's dynamics we can use the methods of 
[dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) such as value iteration and policy iteration
to compute the optimal policy. Todd Neller was the first to do this in his 2004 [paper](https://cupola.gettysburg.edu/cgi/viewcontent.cgi?article=1003&context=csfac).

![Alt text](nellers_optimal_policy.png?raw=true width=50 height=50 "Visualization of the optimal policy for pig from Neller's paper")

### The Purpose of this Work

I am currently working my way through 'Reinforcement Learning' - the classic textbook by Sutton & Barton so this work is
primarily to aid in my understanding of some of the topics there. Pig also has a small place in my heart as one of the 
things that inspired me to begin studying machine learning was a university project centered around discovering
good strategy in Pig - although we were primarily focused on genetic algorithms which looking back are clearly quite
poorly suited to this problem but oh well. I have included a copy of our report from that project in this repo.

My plan is to first reproduce the Neller's optimal policy using both value iteration and policy iteration and try and 
understand the differences between them in terms of their rate of convergence. Neller also discusses some 'tricks' that
can be used to improve the convergence rate by applying value/policy iteration incrementally on disjoint subsets states.
I'd like to analyse what impact these tricks have on convergence - are they required for a reasonable convergence time
or just nice to have?

Moving beyond 'vanilla' pig I'd also like to experiment with variants for which dynamic programming cannot be used. 
This will be the case whenever we no longer have a perfect model of the games dynamics. One idea I have is to simply
simulate a bias dice where the probability of rolling each number is no longer equal and is unknown to the agent. Of course
it'd be quite simple to just have the agent estimate the probability distribution from it's initial experiences and then
apply value/policy iteration on those estimates but where's the fun in that? I'm more interested in seeing if an agent
can learn to approximate the optimal policy through self-play alone.  

  


