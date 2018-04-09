/**
 * @file constants.h
 * @brief Definition of all parameters definition within the program
 */
#ifndef SIM_CONST
#define SIM_CONST


//------------------------------------------------------------------------------
//Learning rewards
//------------------------------------------------------------------------------

#define EAT_ENERGY 0
#define ENTER_ENERGY 0
#define MOVE_ENERGY 0
#define MAX_REWARD 10 //BIGGEST OF THE ABOVE //Needs to be checked with negative rewards
#define MAX_REWARD_DRL 10

//------------------------------------------------------------------------------
//RLearning parameters
//------------------------------------------------------------------------------
#define rl_alpha 0.1
#ifdef TEST
#define rl_gamma 0.1
#else
#define rl_gamma 0.5
#endif
#ifdef TEST
#define rl_epsilon 0.9
#else
#define rl_epsilon 0.1
#endif
#define GAMMA_MULT (1+rl_gamma+(rl_gamma*rl_gamma)+(rl_gamma*rl_gamma*rl_gamma)+(rl_gamma*rl_gamma*rl_gamma*rl_gamma)+(rl_gamma*rl_gamma*rl_gamma*rl_gamma*rl_gamma))

//------------------------------------------------------------------------------
//Learning
//------------------------------------------------------------------------------

#define MAX_INPUT_VAL 100 //MAX perception

#define MAX_WEIGHT ((MAX_REWARD+rl_gamma*MAX_REWARD)/MAX_INPUT_VAL) //This assumes a single layer //truncated

//------------------------------------------------------------------------------
//Population constants
//------------------------------------------------------------------------------

#define SKILL_INCREASE 0.2 //Skill increase < 1
#define LOCK_INTERVAL 50

//! The value to add to the weights when seeding
#define SEED_VALUE 0.3 //MAX_WEIGHT/10.0

//! The following definitions determine how to generate new food, activate only one at a time
#define FOOD_UNIFORM

//------------------------------------------------------------------------------
//Agent constants
//------------------------------------------------------------------------------

//! Set the starting energy and the max random value to be added to it
#ifdef IMMORTALS
#define INIT_ENERGY_CONST 0
#define INIT_ENERGY_VAR 50
#else
// Set accordingly to ENERGY_MAX that is used for reproduction
#define INIT_ENERGY_CONST 0
#define INIT_ENERGY_VAR 50
#endif
/*!< This value sets the range of the noise to be added to the output of the brain, useful to break ties */
#define DECISION_NOISE 0.001*MAX_WEIGHT

//-----------------------------------------------------------------------------
//Genome constants
//------------------------------------------------------------------------------

//#define N_MARKERS 10
/*!< The maximum energy an agent can have. This value drives reproduction.
*/
#define ENERGY_MAX 100

//------------------------------------------------------------------------------
//Field constants
//------------------------------------------------------------------------------

//! Probability used in spawn_bundle()
#define GRASS_SPAWN_PROB 0.0
#define FOOD_TYPES 2
#define FOOD_EXCLUSIVE true

/*!< Define the number of perceptions, needed to initialize the genome
  This number depends on whether markers and food are visible
*/
//TODO use struct constants_t instead
#ifndef INTERACT
#ifndef SEP_FOOD
#define N_PERCEPTIONS 5
#else
#define N_PERCEPTIONS 6
#endif
#elif defined INTERACT
#ifdef INVISIBLE_FOOD
#ifndef SEP_FOOD
#define N_PERCEPTIONS 5
#else
#define N_PERCEPTIONS 7
#endif
#elif !defined INVISIBLE_FOOD
#ifndef SEP_FOOD
#define N_PERCEPTIONS 10
#else
#define N_PERCEPTIONS 11
#endif
#endif //INVISIBLE_FOOD
#endif //INTERACT
#ifdef TEST
#define N_OUTPUTS 3             // move in 2 directions and eat
#else
#define N_OUTPUTS 5             // move in 4 directions and eat
#endif // N_OUTPUTS

#endif
