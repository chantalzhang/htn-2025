import { Athlete } from '@/types';

export const athletes: Athlete[] = [
  {
    id: 'lebron-james',
    name: 'LeBron James',
    sport: 'Basketball',
    position: 'Small Forward',
    height: 206,
    weight: 113,
    wingspan: 214,
    shoulderWidth: 58,
    waist: 86,
    hip: 102,
    imageUrl: '/images/lebron-james.jpg',
    description: '4-time NBA Champion, 4-time Finals MVP',
    achievements: ['4x NBA Champion', '4x Finals MVP', '19x All-Star', '4x MVP']
  },
  {
    id: 'tom-brady',
    name: 'Tom Brady',
    sport: 'Football',
    position: 'Quarterback',
    height: 193,
    weight: 102,
    wingspan: 196,
    shoulderWidth: 55,
    waist: 89,
    hip: 99,
    imageUrl: '/images/tom-brady.jpg',
    description: '7-time Super Bowl Champion',
    achievements: ['7x Super Bowl Champion', '5x Super Bowl MVP', '15x Pro Bowl']
  },
  {
    id: 'lionel-messi',
    name: 'Lionel Messi',
    sport: 'Soccer',
    position: 'Forward',
    height: 170,
    weight: 72,
    wingspan: 175,
    shoulderWidth: 45,
    waist: 76,
    hip: 88,
    imageUrl: '/images/lionel-messi.jpg',
    description: '8-time Ballon d\'Or winner',
    achievements: ['8x Ballon d\'Or', 'World Cup Champion', '4x Champions League']
  },
  {
    id: 'serena-williams',
    name: 'Serena Williams',
    sport: 'Tennis',
    position: 'Singles',
    height: 175,
    weight: 70,
    wingspan: 187,
    shoulderWidth: 48,
    waist: 71,
    hip: 91,
    imageUrl: '/images/serena-williams.jpg',
    description: '23-time Grand Slam Champion',
    achievements: ['23x Grand Slam Singles', '14x Grand Slam Doubles', '4x Olympic Gold']
  },
  {
    id: 'usain-bolt',
    name: 'Usain Bolt',
    sport: 'Track & Field',
    position: 'Sprinter',
    height: 195,
    weight: 94,
    wingspan: 200,
    shoulderWidth: 52,
    waist: 81,
    hip: 95,
    imageUrl: '/images/usain-bolt.jpg',
    description: 'Fastest man in the world',
    achievements: ['8x Olympic Gold', '11x World Champion', 'World Record Holder']
  },
  {
    id: 'michael-phelps',
    name: 'Michael Phelps',
    sport: 'Swimming',
    position: 'Multiple',
    height: 193,
    weight: 88,
    wingspan: 203,
    shoulderWidth: 56,
    waist: 79,
    hip: 92,
    imageUrl: '/images/michael-phelps.jpg',
    description: 'Most decorated Olympian of all time',
    achievements: ['23x Olympic Gold', '28x Olympic Medal', '39x World Record']
  },
  {
    id: 'conor-mcgregor',
    name: 'Conor McGregor',
    sport: 'MMA',
    position: 'Lightweight/Welterweight',
    height: 175,
    weight: 70,
    wingspan: 188,
    shoulderWidth: 47,
    waist: 76,
    hip: 89,
    imageUrl: '/images/conor-mcgregor.jpg',
    description: 'Former UFC Featherweight and Lightweight Champion',
    achievements: ['UFC Featherweight Champion', 'UFC Lightweight Champion', 'First UFC Double Champion']
  },
  {
    id: 'simone-biles',
    name: 'Simone Biles',
    sport: 'Gymnastics',
    position: 'All-Around',
    height: 142,
    weight: 47,
    wingspan: 150,
    shoulderWidth: 35,
    waist: 61,
    hip: 76,
    imageUrl: '/images/simone-biles.jpg',
    description: 'Most decorated gymnast in World Championships history',
    achievements: ['7x Olympic Medal', '25x World Championship Medal', '4x Olympic Gold']
  },
  {
    id: 'cristiano-ronaldo',
    name: 'Cristiano Ronaldo',
    sport: 'Soccer',
    position: 'Forward',
    height: 187,
    weight: 83,
    wingspan: 192,
    shoulderWidth: 50,
    waist: 81,
    hip: 94,
    imageUrl: '/images/cristiano-ronaldo.jpg',
    description: '5-time Ballon d\'Or winner',
    achievements: ['5x Ballon d\'Or', '5x Champions League', 'European Championship']
  },
  {
    id: 'stephen-curry',
    name: 'Stephen Curry',
    sport: 'Basketball',
    position: 'Point Guard',
    height: 188,
    weight: 86,
    wingspan: 192,
    shoulderWidth: 48,
    waist: 79,
    hip: 91,
    imageUrl: '/images/stephen-curry.jpg',
    description: '4-time NBA Champion, Greatest shooter of all time',
    achievements: ['4x NBA Champion', '2x MVP', '9x All-Star', '3-Point Record Holder']
  },
  {
    id: 'alex-morgan',
    name: 'Alex Morgan',
    sport: 'Soccer',
    position: 'Forward',
    height: 170,
    weight: 59,
    wingspan: 175,
    shoulderWidth: 42,
    waist: 66,
    hip: 81,
    imageUrl: '/images/alex-morgan.jpg',
    description: '2-time World Cup Champion',
    achievements: ['2x World Cup Champion', 'Olympic Gold Medal', 'NWSL Champion']
  },
  {
    id: 'kobe-bryant',
    name: 'Kobe Bryant',
    sport: 'Basketball',
    position: 'Shooting Guard',
    height: 198,
    weight: 96,
    wingspan: 211,
    shoulderWidth: 54,
    waist: 84,
    hip: 98,
    imageUrl: '/images/kobe-bryant.jpg',
    description: '5-time NBA Champion, Black Mamba',
    achievements: ['5x NBA Champion', '2x Finals MVP', '18x All-Star', '2x Olympic Gold']
  }
];

export const sports = [
  {
    name: 'Basketball',
    icon: '🏀',
    description: 'High-intensity team sport requiring height, agility, and endurance',
    keyTraits: ['height', 'wingspan', 'agility', 'endurance']
  },
  {
    name: 'Football',
    icon: '🏈',
    description: 'Physical team sport with various positions requiring different body types',
    keyTraits: ['strength', 'speed', 'size', 'power']
  },
  {
    name: 'Soccer',
    icon: '⚽',
    description: 'Endurance-based team sport requiring speed, agility, and stamina',
    keyTraits: ['endurance', 'speed', 'agility', 'stamina']
  },
  {
    name: 'Tennis',
    icon: '🎾',
    description: 'Individual sport requiring power, precision, and mental toughness',
    keyTraits: ['power', 'precision', 'endurance', 'mental_toughness']
  },
  {
    name: 'Track & Field',
    icon: '🏃',
    description: 'Speed and power events requiring explosive athleticism',
    keyTraits: ['speed', 'power', 'explosiveness', 'technique']
  },
  {
    name: 'Swimming',
    icon: '🏊',
    description: 'Full-body endurance sport requiring technique and lung capacity',
    keyTraits: ['endurance', 'technique', 'lung_capacity', 'flexibility']
  },
  {
    name: 'MMA',
    icon: '🥊',
    description: 'Combat sport requiring strength, technique, and mental toughness',
    keyTraits: ['strength', 'technique', 'mental_toughness', 'flexibility']
  },
  {
    name: 'Gymnastics',
    icon: '🤸',
    description: 'Precision sport requiring flexibility, strength, and body control',
    keyTraits: ['flexibility', 'strength', 'body_control', 'precision']
  }
];
