% Факты с одним аргументом (персонажи)
character('Batman').
character('Alfred Pennyworth').
character('Joker').
character('Harley Quinn').
character('Scarecrow').
character('Nightwing').
character('Robin').
character('Catwoman').
character('Arkham Knight').
character('Two-Face').
character('Penguin').
character('Riddler').
character('Poison Ivy').
character('Bane').
character('Oracle').
character('Jim Gordon').
character('Deathstroke').

%Типы героев

type('Hero').
type('Ally').
type('Villain').
type('Antihero').


% Факты с двумя аргументами 
role('Batman', 'Hero').
role('Alfred Pennyworth', 'Ally').
role('Joker', 'Villain').
role('Harley Quinn', 'Villain').
role('Scarecrow', 'Villain').
role('Nightwing', 'Hero').
role('Robin', 'Hero').
role('Catwoman', 'Antihero').
role('Arkham Knight', 'Villain').
role('Two-Face', 'Villain').
role('Penguin', 'Villain').
role('Riddler', 'Villain').
role('Poison Ivy', 'Villain').
role('Bane', 'Villain').
role('Oracle', 'Ally').
role('Jim Gordon', 'Ally').
role('Deathstroke', 'Villain').

%Факты о поле
sex('Batman', 'M').
sex('Alfred Pennyworth', 'M').
sex('Joker', 'M').
sex('Harley Quinn', 'F').
sex('Scarecrow', 'M').
sex('Nightwing', 'M').
sex('Robin', 'M').
sex('Catwoman', 'F').
sex('Arkham Knight', 'M').
sex('Two-Face', 'M').
sex('Penguin', 'M').
sex('Riddler', 'M').
sex('Poison Ivy', 'F').
sex('Bane', 'M').
sex('Oracle', 'F').
sex('Jim Gordon', 'M').
sex('Deathstroke', 'M').

% Факты о взаимодействиях
alliance('Batman', 'Alfred Pennyworth').
alliance('Batman', 'Nightwing').
alliance('Batman', 'Robin').
alliance('Batman', 'Catwoman').
alliance('Alfred Pennyworth', 'Oracle').
alliance('Jim Gordon', 'Batman').

% Правила
is_hero(Character) :- character(Character), role(Character, 'Hero').
is_villain(Character) :- character(Character), role(Character, 'Villain').
is_antihero(Character) :- character(Character), role(Character, 'Antihero').
is_ally(Character) :- character(Character), role(Character, 'Ally').

is_enemy(Character1, Character2) :- 
    is_hero(Character1), is_villain(Character2).

is_allied(Character1, Character2) :- 
    alliance(Character1, Character2),Character1 \= Character2.

% Правило для определения конфликтов
has_conflict(Character1, Character2) :- 
    (is_hero(Character1), is_villain(Character2));
    (is_villain(Character1), is_hero(Character2)).

% Правило для определения, кто может помочь
can_help(Character1, Character2) :- 
    is_ally(Character1), (is_hero(Character2); is_antihero(Character2)).

% Правило для определения, может ли персонаж участвовать в битве
can_fight(Character1, Character2) :- 
    has_conflict(Character1, Character2), 
    (is_hero(Character1); is_villain(Character1)).

% Правило для определения, является ли персонаж опасным
is_dangerous(Character) :- (is_villain(Character), (Character == 'Joker'; Character == 'Scarecrow'; Character == 'Bane'));(is_hero(Character),Character == 'Batman').

% Правило для определения, может ли персонаж получить помощь
needs_help(Character) :-   is_hero(Character), not(is_allied(Character, 'Batman')).

% Правило для определения, может ли персонаж обмануть
can_trick(Character) :- 
    (is_villain(Character), (Character == 'Harley Quinn'; Character == 'Riddler'));(is_hero(Character),Character == 'Robin').

% Правило для определения, кто является основным антагонистом
main_antagonist(Antagonist) :- is_villain(Antagonist), ( Antagonist == 'Arkham Knight';Antagonist == 'Scarecrow').


% Пример запросов:
% ?- is_hero('Batman'). % true
% ?- is_villain('Joker'). % true

% ?- is_villain('Joker'), is_dangerous('Joker'). % true
% ?- (is_villain('Harley Quinn'); is_ally('Harley Quinn')). % true

% ?- is_villain(Villain).
% ?- is_allied('Batman', Ally).
% ?- can_fight('Batman', Enemy).

% ?- can_trick(Trickster).
% ?- needs_help(Character).
