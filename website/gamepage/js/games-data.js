// Games database - sourced from Internet Archive and other legal sources
// Each game has: id, title, description, genres (array), year, screenshot, archiveUrl

const GAMES_DATABASE = [
    {
        id: "doom",
        title: "DOOM",
        description: "The legendary first-person shooter that defined a genre. Fight through hordes of demons from Hell in this groundbreaking 1993 classic by id Software.",
        genres: ["action", "fps", "horror"],
        year: 1993,
        screenshot: "https://archive.org/services/img/msdos_DOOM_1993",
        archiveId: "msdos_DOOM_1993"
    },
    {
        id: "doom2",
        title: "DOOM II: Hell on Earth",
        description: "The sequel to DOOM. Earth has been invaded by demons and you're humanity's last hope. Features the iconic Super Shotgun.",
        genres: ["action", "fps", "horror"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_DOOM_II_-_Hell_on_Earth_1994",
        archiveId: "msdos_DOOM_II_-_Hell_on_Earth_1994"
    },
    {
        id: "quake",
        title: "Quake",
        description: "id Software's groundbreaking 3D shooter. Battle Lovecraftian monsters through dark medieval environments with true 3D graphics.",
        genres: ["action", "fps", "horror"],
        year: 1996,
        screenshot: "https://archive.org/services/img/msdos_Quake_1996",
        archiveId: "msdos_Quake_1996"
    },
    {
        id: "duke3d",
        title: "Duke Nukem 3D",
        description: "The wise-cracking action hero Duke Nukem battles alien invaders in this irreverent FPS classic. Hail to the king, baby!",
        genres: ["action", "fps"],
        year: 1996,
        screenshot: "https://archive.org/services/img/msdos_Duke_Nukem_3D_1996",
        archiveId: "msdos_Duke_Nukem_3D_1996"
    },
    {
        id: "wolfenstein3d",
        title: "Wolfenstein 3D",
        description: "The grandfather of FPS games. Escape from Castle Wolfenstein, fighting Nazi soldiers and eventually Hitler himself.",
        genres: ["action", "fps"],
        year: 1992,
        screenshot: "https://archive.org/services/img/msdos_Wolfenstein_3D_1992",
        archiveId: "msdos_Wolfenstein_3D_1992"
    },
    {
        id: "commander-keen-4",
        title: "Commander Keen 4: Secret of the Oracle",
        description: "Billy Blaze, an 8-year-old genius, dons his brother's football helmet and saves the galaxy as Commander Keen in this classic platformer.",
        genres: ["platformer", "action", "adventure"],
        year: 1991,
        screenshot: "https://archive.org/services/img/msdos_Commander_Keen_4_-_Secret_of_the_Oracle_1991",
        archiveId: "msdos_Commander_Keen_4_-_Secret_of_the_Oracle_1991"
    },
    {
        id: "prince-of-persia",
        title: "Prince of Persia",
        description: "The groundbreaking cinematic platformer. You have 60 minutes to escape the dungeon and rescue the Princess from the evil Jaffar.",
        genres: ["platformer", "action", "adventure"],
        year: 1989,
        screenshot: "https://archive.org/services/img/msdos_Prince_of_Persia_1990",
        archiveId: "msdos_Prince_of_Persia_1990"
    },
    {
        id: "prince-of-persia-2",
        title: "Prince of Persia 2: The Shadow and the Flame",
        description: "The sequel to the groundbreaking platformer. The Prince is framed and must flee the palace, embarking on a new adventure.",
        genres: ["platformer", "action", "adventure"],
        year: 1993,
        screenshot: "https://archive.org/services/img/msdos_Prince_of_Persia_2_-_The_Shadow_and_the_Flame_1993",
        archiveId: "msdos_Prince_of_Persia_2_-_The_Shadow_and_the_Flame_1993"
    },
    {
        id: "oregon-trail",
        title: "The Oregon Trail",
        description: "Lead your pioneer family on the perilous journey from Missouri to Oregon. Hunt for food, ford rivers, and try not to die of dysentery.",
        genres: ["strategy", "simulation", "educational"],
        year: 1990,
        screenshot: "https://archive.org/services/img/msdos_Oregon_Trail_The_1990",
        archiveId: "msdos_Oregon_Trail_The_1990"
    },
    {
        id: "simcity",
        title: "SimCity",
        description: "Build and manage your own city. Zone residential, commercial, and industrial areas, manage utilities, and deal with disasters.",
        genres: ["simulation", "strategy"],
        year: 1989,
        screenshot: "https://archive.org/services/img/msdos_SimCity_1989",
        archiveId: "msdos_SimCity_1989"
    },
    {
        id: "simcity-2000",
        title: "SimCity 2000",
        description: "The enhanced sequel to SimCity with isometric graphics, underground layers, and more complex city management.",
        genres: ["simulation", "strategy"],
        year: 1993,
        screenshot: "https://archive.org/services/img/msdos_SimCity_2000_1993",
        archiveId: "msdos_SimCity_2000_1993"
    },
    {
        id: "civilization",
        title: "Sid Meier's Civilization",
        description: "Build an empire to stand the test of time. Guide your civilization from the ancient era to the space age in this legendary strategy game.",
        genres: ["strategy", "simulation"],
        year: 1991,
        screenshot: "https://archive.org/services/img/msdos_Sid_Meiers_Civilization_1991",
        archiveId: "msdos_Sid_Meiers_Civilization_1991"
    },
    {
        id: "warcraft",
        title: "Warcraft: Orcs & Humans",
        description: "The game that started it all. Command the forces of either Humans or Orcs in this real-time strategy classic from Blizzard.",
        genres: ["strategy", "rts"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_Warcraft_-_Orcs__Humans_1994",
        archiveId: "msdos_Warcraft_-_Orcs__Humans_1994"
    },
    {
        id: "warcraft2",
        title: "Warcraft II: Tides of Darkness",
        description: "The epic sequel with naval combat, improved graphics, and the iconic voice acting. For the Horde! For the Alliance!",
        genres: ["strategy", "rts"],
        year: 1995,
        screenshot: "https://archive.org/services/img/msdos_Warcraft_II_-_Tides_of_Darkness_1995",
        archiveId: "msdos_Warcraft_II_-_Tides_of_Darkness_1995"
    },
    {
        id: "command-conquer",
        title: "Command & Conquer",
        description: "GDI vs NOD in the battle for Tiberium. Westwood's groundbreaking RTS that defined the genre with its FMV cutscenes.",
        genres: ["strategy", "rts"],
        year: 1995,
        screenshot: "https://archive.org/services/img/msdos_Command__Conquer_1995",
        archiveId: "msdos_Command__Conquer_1995"
    },
    {
        id: "red-alert",
        title: "Command & Conquer: Red Alert",
        description: "Einstein goes back in time to prevent WWII, creating an alternate history where Soviets invade Europe. Kirov reporting!",
        genres: ["strategy", "rts"],
        year: 1996,
        screenshot: "https://archive.org/services/img/msdos_Command__Conquer_-_Red_Alert_1996",
        archiveId: "msdos_Command__Conquer_-_Red_Alert_1996"
    },
    {
        id: "lemmings",
        title: "Lemmings",
        description: "Guide hordes of mindless Lemmings to safety using their special abilities. A puzzle classic that spawned countless imitators.",
        genres: ["puzzle", "strategy"],
        year: 1991,
        screenshot: "https://archive.org/services/img/msdos_Lemmings_1991",
        archiveId: "msdos_Lemmings_1991"
    },
    {
        id: "tetris",
        title: "Tetris",
        description: "The legendary puzzle game from Russia. Stack falling tetrominoes to clear lines in this endlessly addictive classic.",
        genres: ["puzzle"],
        year: 1986,
        screenshot: "https://archive.org/services/img/msdos_Tetris_1986",
        archiveId: "msdos_Tetris_1986"
    },
    {
        id: "monkey-island",
        title: "The Secret of Monkey Island",
        description: "Guybrush Threepwood wants to be a pirate! LucasArts' hilarious point-and-click adventure with unforgettable humor and insult sword fighting.",
        genres: ["adventure", "puzzle"],
        year: 1990,
        screenshot: "https://archive.org/services/img/msdos_Secret_of_Monkey_Island_The_1990",
        archiveId: "msdos_Secret_of_Monkey_Island_The_1990"
    },
    {
        id: "monkey-island-2",
        title: "Monkey Island 2: LeChuck's Revenge",
        description: "Guybrush returns to find Big Whoop treasure while the ghost pirate LeChuck hunts him. Even funnier than the original!",
        genres: ["adventure", "puzzle"],
        year: 1991,
        screenshot: "https://archive.org/services/img/msdos_Monkey_Island_2_-_LeChucks_Revenge_1991",
        archiveId: "msdos_Monkey_Island_2_-_LeChucks_Revenge_1991"
    },
    {
        id: "day-of-tentacle",
        title: "Day of the Tentacle",
        description: "Save the world from Purple Tentacle by sending three friends through time. One of the funniest adventure games ever made.",
        genres: ["adventure", "puzzle"],
        year: 1993,
        screenshot: "https://archive.org/services/img/msdos_Maniac_Mansion_-_Day_of_the_Tentacle_1993",
        archiveId: "msdos_Maniac_Mansion_-_Day_of_the_Tentacle_1993"
    },
    {
        id: "indiana-jones-fate",
        title: "Indiana Jones and the Fate of Atlantis",
        description: "Join Indy on an adventure to find the lost city of Atlantis before the Nazis. Multiple paths to complete the game!",
        genres: ["adventure", "puzzle"],
        year: 1992,
        screenshot: "https://archive.org/services/img/msdos_Indiana_Jones_and_the_Fate_of_Atlantis_1992",
        archiveId: "msdos_Indiana_Jones_and_the_Fate_of_Atlantis_1992"
    },
    {
        id: "sam-and-max",
        title: "Sam & Max Hit the Road",
        description: "A dog and rabbit detective duo investigate a missing bigfoot in this hilarious LucasArts adventure.",
        genres: ["adventure", "puzzle"],
        year: 1993,
        screenshot: "https://archive.org/services/img/msdos_Sam__Max_-_Hit_the_Road_1993",
        archiveId: "msdos_Sam__Max_-_Hit_the_Road_1993"
    },
    {
        id: "space-quest-4",
        title: "Space Quest IV: Roger Wilco and The Time Rippers",
        description: "Roger Wilco travels through time and parodies other Space Quest games in this hilarious Sierra adventure.",
        genres: ["adventure", "puzzle", "sci-fi"],
        year: 1991,
        screenshot: "https://archive.org/services/img/msdos_Space_Quest_IV_-_Roger_Wilco_and_the_Time_Rippers_1991",
        archiveId: "msdos_Space_Quest_IV_-_Roger_Wilco_and_the_Time_Rippers_1991"
    },
    {
        id: "kings-quest-6",
        title: "King's Quest VI: Heir Today, Gone Tomorrow",
        description: "Prince Alexander searches for Princess Cassima across the Land of the Green Isles. Considered the best in the series.",
        genres: ["adventure", "puzzle", "fantasy"],
        year: 1992,
        screenshot: "https://archive.org/services/img/msdos_Kings_Quest_VI_-_Heir_Today_Gone_Tomorrow_1992",
        archiveId: "msdos_Kings_Quest_VI_-_Heir_Today_Gone_Tomorrow_1992"
    },
    {
        id: "ultima-7",
        title: "Ultima VII: The Black Gate",
        description: "The Avatar returns to Britannia to investigate a ritualistic murder and uncover a sinister plot. An RPG masterpiece.",
        genres: ["rpg", "adventure", "fantasy"],
        year: 1992,
        screenshot: "https://archive.org/services/img/msdos_Ultima_VII_-_The_Black_Gate_1992",
        archiveId: "msdos_Ultima_VII_-_The_Black_Gate_1992"
    },
    {
        id: "eye-of-beholder",
        title: "Eye of the Beholder",
        description: "A first-person dungeon crawl through the sewers beneath Waterdeep. Real-time combat in the Dungeons & Dragons universe.",
        genres: ["rpg", "adventure", "fantasy"],
        year: 1991,
        screenshot: "https://archive.org/services/img/msdos_Eye_of_the_Beholder_1991",
        archiveId: "msdos_Eye_of_the_Beholder_1991"
    },
    {
        id: "betrayal-at-krondor",
        title: "Betrayal at Krondor",
        description: "An epic RPG set in Raymond E. Feist's Riftwar universe. Deep story, tactical combat, and a vast world to explore.",
        genres: ["rpg", "adventure", "fantasy"],
        year: 1993,
        screenshot: "https://archive.org/services/img/msdos_Betrayal_at_Krondor_1993",
        archiveId: "msdos_Betrayal_at_Krondor_1993"
    },
    {
        id: "xcom-ufo-defense",
        title: "X-COM: UFO Defense",
        description: "Defend Earth from alien invasion in this legendary tactical strategy game. Build bases, research tech, and lead your soldiers.",
        genres: ["strategy", "rpg", "sci-fi"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_X-COM_-_UFO_Defense_1994",
        archiveId: "msdos_X-COM_-_UFO_Defense_1994"
    },
    {
        id: "xcom-terror",
        title: "X-COM: Terror from the Deep",
        description: "The aliens attack from the ocean depths. Underwater combat and new terrors in this challenging sequel to UFO Defense.",
        genres: ["strategy", "rpg", "sci-fi"],
        year: 1995,
        screenshot: "https://archive.org/services/img/msdos_X-COM_-_Terror_from_the_Deep_1995",
        archiveId: "msdos_X-COM_-_Terror_from_the_Deep_1995"
    },
    {
        id: "aladdin",
        title: "Disney's Aladdin",
        description: "Based on the Disney film, swing swords, throw apples, and magic carpet ride through Agrabah in this gorgeous platformer.",
        genres: ["platformer", "action"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_Aladdin_1994",
        archiveId: "msdos_Aladdin_1994"
    },
    {
        id: "lion-king",
        title: "The Lion King",
        description: "Play as Simba from cub to king in this challenging Disney platformer with beautiful animation.",
        genres: ["platformer", "action"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_Lion_King_The_1994",
        archiveId: "msdos_Lion_King_The_1994"
    },
    {
        id: "jazz-jackrabbit",
        title: "Jazz Jackrabbit",
        description: "Epic Games' answer to Sonic. Jazz the rabbit blasts through colorful levels at high speed to save the princess.",
        genres: ["platformer", "action"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_Jazz_Jackrabbit_1994",
        archiveId: "msdos_Jazz_Jackrabbit_1994"
    },
    {
        id: "raptor",
        title: "Raptor: Call of the Shadows",
        description: "A mercenary pilot takes on missions for MegaCorp in this smooth-scrolling vertical shooter with great graphics.",
        genres: ["action", "shooter"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_Raptor_-_Call_of_the_Shadows_1994",
        archiveId: "msdos_Raptor_-_Call_of_the_Shadows_1994"
    },
    {
        id: "tyrian",
        title: "Tyrian 2000",
        description: "One of the greatest shoot-em-ups ever made. Now freeware! Incredible weapon variety and secrets.",
        genres: ["action", "shooter"],
        year: 1999,
        screenshot: "https://archive.org/services/img/msdos_Tyrian_2000_1999",
        archiveId: "msdos_Tyrian_2000_1999"
    },
    {
        id: "one-must-fall",
        title: "One Must Fall: 2097",
        description: "Giant robot fighting game with RPG elements. Upgrade your robot and pilot between matches.",
        genres: ["action", "fighting"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_One_Must_Fall_2097_1994",
        archiveId: "msdos_One_Must_Fall_2097_1994"
    },
    {
        id: "mortal-kombat",
        title: "Mortal Kombat",
        description: "The controversial fighting game that started it all. Fatalities, blood, and kombat!",
        genres: ["action", "fighting"],
        year: 1993,
        screenshot: "https://archive.org/services/img/msdos_Mortal_Kombat_1993",
        archiveId: "msdos_Mortal_Kombat_1993"
    },
    {
        id: "mortal-kombat-2",
        title: "Mortal Kombat II",
        description: "The improved sequel with more characters, more fatalities, and babalities. TOASTY!",
        genres: ["action", "fighting"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_Mortal_Kombat_II_1994",
        archiveId: "msdos_Mortal_Kombat_II_1994"
    },
    {
        id: "street-fighter-2",
        title: "Street Fighter II",
        description: "The legendary fighting game that defined the genre. Hadouken your way to victory!",
        genres: ["action", "fighting"],
        year: 1992,
        screenshot: "https://archive.org/services/img/msdos_Street_Fighter_II_1992",
        archiveId: "msdos_Street_Fighter_II_1992"
    },
    {
        id: "dune-2",
        title: "Dune II: The Building of a Dynasty",
        description: "The game that invented real-time strategy. Harvest spice, build your base, and crush the enemy houses on Arrakis.",
        genres: ["strategy", "rts", "sci-fi"],
        year: 1992,
        screenshot: "https://archive.org/services/img/msdos_Dune_II_-_The_Building_of_a_Dynasty_1992",
        archiveId: "msdos_Dune_II_-_The_Building_of_a_Dynasty_1992"
    },
    {
        id: "heroes-mm2",
        title: "Heroes of Might and Magic II",
        description: "Build your kingdom, recruit heroes, and conquer the land in this beloved turn-based strategy game.",
        genres: ["strategy", "rpg", "fantasy"],
        year: 1996,
        screenshot: "https://archive.org/services/img/msdos_Heroes_of_Might_and_Magic_II_-_The_Succession_Wars_1996",
        archiveId: "msdos_Heroes_of_Might_and_Magic_II_-_The_Succession_Wars_1996"
    },
    {
        id: "master-of-orion",
        title: "Master of Orion",
        description: "4X space strategy at its finest. Explore, expand, exploit, and exterminate your way to galactic domination.",
        genres: ["strategy", "sci-fi"],
        year: 1993,
        screenshot: "https://archive.org/services/img/msdos_Master_of_Orion_1993",
        archiveId: "msdos_Master_of_Orion_1993"
    },
    {
        id: "master-of-orion-2",
        title: "Master of Orion II: Battle at Antares",
        description: "The legendary space 4X sequel. Ship design, tactical combat, and galaxy conquest.",
        genres: ["strategy", "sci-fi"],
        year: 1996,
        screenshot: "https://archive.org/services/img/msdos_Master_of_Orion_II_-_Battle_at_Antares_1996",
        archiveId: "msdos_Master_of_Orion_II_-_Battle_at_Antares_1996"
    },
    {
        id: "transport-tycoon",
        title: "Transport Tycoon Deluxe",
        description: "Build a transportation empire with trains, trucks, ships, and planes. The original logistics management game.",
        genres: ["simulation", "strategy"],
        year: 1995,
        screenshot: "https://archive.org/services/img/msdos_Transport_Tycoon_Deluxe_1995",
        archiveId: "msdos_Transport_Tycoon_Deluxe_1995"
    },
    {
        id: "theme-hospital",
        title: "Theme Hospital",
        description: "Build and manage a hospital, cure bizarre diseases, and try not to let too many patients die. Darkly funny management sim.",
        genres: ["simulation", "strategy"],
        year: 1997,
        screenshot: "https://archive.org/services/img/msdos_Theme_Hospital_1997",
        archiveId: "msdos_Theme_Hospital_1997"
    },
    {
        id: "theme-park",
        title: "Theme Park",
        description: "Build the ultimate amusement park. Design roller coasters, set prices, and watch the money roll in.",
        genres: ["simulation", "strategy"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_Theme_Park_1994",
        archiveId: "msdos_Theme_Park_1994"
    },
    {
        id: "carmageddon",
        title: "Carmageddon",
        description: "Win races by crossing the finish line, wrecking opponents, or running over every pedestrian. Controversial and fun.",
        genres: ["racing", "action"],
        year: 1997,
        screenshot: "https://archive.org/services/img/msdos_Carmageddon_1997",
        archiveId: "msdos_Carmageddon_1997"
    },
    {
        id: "need-for-speed",
        title: "Need for Speed",
        description: "The racing franchise that started it all. Race exotic cars on scenic roads while evading police.",
        genres: ["racing"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_Need_for_Speed_The_1994",
        archiveId: "msdos_Need_for_Speed_The_1994"
    },
    {
        id: "wacky-wheels",
        title: "Wacky Wheels",
        description: "Mario Kart for DOS! Race as cute animals, collect power-ups, and battle your way to first place.",
        genres: ["racing", "action"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_Wacky_Wheels_1994",
        archiveId: "msdos_Wacky_Wheels_1994"
    },
    {
        id: "blood",
        title: "Blood",
        description: "One of the Build engine's darkest games. Caleb rises from the grave to seek revenge against the dark god Tchernobog.",
        genres: ["action", "fps", "horror"],
        year: 1997,
        screenshot: "https://archive.org/services/img/msdos_Blood_1997",
        archiveId: "msdos_Blood_1997"
    },
    {
        id: "shadow-warrior",
        title: "Shadow Warrior",
        description: "Lo Wang takes on demons with katanas and crude humor in this Build engine shooter.",
        genres: ["action", "fps"],
        year: 1997,
        screenshot: "https://archive.org/services/img/msdos_Shadow_Warrior_1997",
        archiveId: "msdos_Shadow_Warrior_1997"
    },
    {
        id: "rise-of-triad",
        title: "Rise of the Triad",
        description: "Insane weapons, ludicrous gibs, and dog mode. An underrated FPS gem from Apogee.",
        genres: ["action", "fps"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_Rise_of_the_Triad_-_Dark_War_1994",
        archiveId: "msdos_Rise_of_the_Triad_-_Dark_War_1994"
    },
    {
        id: "heretic",
        title: "Heretic",
        description: "DOOM with a fantasy twist. Use magic artifacts and medieval weapons against the undead legions.",
        genres: ["action", "fps", "fantasy"],
        year: 1994,
        screenshot: "https://archive.org/services/img/msdos_Heretic_1994",
        archiveId: "msdos_Heretic_1994"
    },
    {
        id: "hexen",
        title: "Hexen: Beyond Heretic",
        description: "Choose from three character classes and battle through a dark fantasy world. Hub-based level design.",
        genres: ["action", "fps", "fantasy"],
        year: 1995,
        screenshot: "https://archive.org/services/img/msdos_Hexen_-_Beyond_Heretic_1995",
        archiveId: "msdos_Hexen_-_Beyond_Heretic_1995"
    }
];

// All available genres
const ALL_GENRES = [
    { id: "action", name: "Action" },
    { id: "adventure", name: "Adventure" },
    { id: "fps", name: "FPS" },
    { id: "rpg", name: "RPG" },
    { id: "strategy", name: "Strategy" },
    { id: "rts", name: "RTS" },
    { id: "simulation", name: "Simulation" },
    { id: "puzzle", name: "Puzzle" },
    { id: "platformer", name: "Platformer" },
    { id: "racing", name: "Racing" },
    { id: "fighting", name: "Fighting" },
    { id: "shooter", name: "Shooter" },
    { id: "horror", name: "Horror" },
    { id: "fantasy", name: "Fantasy" },
    { id: "sci-fi", name: "Sci-Fi" },
    { id: "educational", name: "Educational" }
];

// Helper functions for game data
function getGameById(id) {
    return GAMES_DATABASE.find(game => game.id === id);
}

function getGamesByGenre(genre) {
    if (genre === 'all') return GAMES_DATABASE;
    return GAMES_DATABASE.filter(game => game.genres.includes(genre));
}

function searchGames(query) {
    const lowerQuery = query.toLowerCase();
    return GAMES_DATABASE.filter(game =>
        game.title.toLowerCase().includes(lowerQuery) ||
        game.description.toLowerCase().includes(lowerQuery) ||
        game.genres.some(g => g.includes(lowerQuery))
    );
}

function getRandomGame() {
    return GAMES_DATABASE[Math.floor(Math.random() * GAMES_DATABASE.length)];
}

function getRandomGames(count) {
    const shuffled = [...GAMES_DATABASE].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
}
