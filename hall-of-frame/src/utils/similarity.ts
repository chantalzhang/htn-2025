import { BodyMeasurements, Athlete, SportRecommendation, AthleteMatch, BodyPart } from '@/types';
import { athletes, sports } from '@/data/athletes';

export function calculateSimilarity(user: BodyMeasurements, athlete: Athlete): number {
  const weights = {
    height: 0.3,
    weight: 0.25,
    wingspan: 0.25,
    shoulderWidth: 0.1,
    waist: 0.05,
    hip: 0.05
  };

  let totalScore = 0;
  let totalWeight = 0;

  // Calculate similarity for each measurement
  Object.entries(weights).forEach(([key, weight]) => {
    const userValue = user[key as keyof BodyMeasurements];
    const athleteValue = athlete[key as keyof Athlete];
    
    if (userValue && athleteValue && typeof userValue === 'number' && typeof athleteValue === 'number') {
      const similarity = 1 - Math.abs(userValue - athleteValue) / Math.max(userValue, athleteValue);
      totalScore += similarity * weight;
      totalWeight += weight;
    }
  });

  return totalWeight > 0 ? (totalScore / totalWeight) * 100 : 0;
}

export function findTopAthleteMatches(user: BodyMeasurements, limit: number = 3): AthleteMatch[] {
  const matches = athletes.map(athlete => ({
    athlete,
    similarityScore: calculateSimilarity(user, athlete),
    matchingTraits: getMatchingTraits(user, athlete)
  }));

  return matches
    .sort((a, b) => b.similarityScore - a.similarityScore)
    .slice(0, limit);
}

export function getMatchingTraits(user: BodyMeasurements, athlete: Athlete): string[] {
  const traits: string[] = [];
  const tolerance = 0.1; // 10% tolerance

  if (Math.abs(user.height - athlete.height) / athlete.height < tolerance) {
    traits.push('Height');
  }
  if (Math.abs(user.weight - athlete.weight) / athlete.weight < tolerance) {
    traits.push('Weight');
  }
  if (Math.abs(user.wingspan - athlete.wingspan) / athlete.wingspan < tolerance) {
    traits.push('Wingspan');
  }
  if (user.shoulderWidth && athlete.shoulderWidth && 
      Math.abs(user.shoulderWidth - athlete.shoulderWidth) / athlete.shoulderWidth < tolerance) {
    traits.push('Shoulder Width');
  }

  return traits;
}

export function recommendSports(user: BodyMeasurements): SportRecommendation[] {
  const recommendations: SportRecommendation[] = [];

  sports.forEach(sport => {
    let score = 0;
    const whyMatch: string[] = [];

    // Basketball scoring
    if (sport.name === 'Basketball') {
      if (user.height > 190) {
        score += 30;
        whyMatch.push('Your height gives you a significant advantage');
      }
      if (user.wingspan > user.height + 5) {
        score += 25;
        whyMatch.push('Your long wingspan is perfect for defense and rebounding');
      }
      if (user.weight > 80 && user.weight < 120) {
        score += 20;
        whyMatch.push('Your weight is ideal for basketball performance');
      }
      if (user.height > 180) {
        score += 25;
        whyMatch.push('Your height is suitable for most basketball positions');
      }
    }

    // Football scoring
    if (sport.name === 'Football') {
      if (user.weight > 90) {
        score += 25;
        whyMatch.push('Your weight provides good power and strength');
      }
      if (user.height > 180) {
        score += 20;
        whyMatch.push('Your height is advantageous for many positions');
      }
      if (user.wingspan > user.height) {
        score += 15;
        whyMatch.push('Your reach is beneficial for catching and blocking');
      }
      if (user.weight > 100) {
        score += 20;
        whyMatch.push('Your size is ideal for lineman positions');
      }
      if (user.height > 190 && user.weight < 110) {
        score += 20;
        whyMatch.push('Your build is perfect for skill positions');
      }
    }

    // Soccer scoring
    if (sport.name === 'Soccer') {
      if (user.height > 170 && user.height < 190) {
        score += 30;
        whyMatch.push('Your height is ideal for soccer');
      }
      if (user.weight > 60 && user.weight < 85) {
        score += 25;
        whyMatch.push('Your weight is perfect for endurance and agility');
      }
      if (user.height < 180) {
        score += 20;
        whyMatch.push('Your height is great for quick movements');
      }
      if (user.weight < 80) {
        score += 25;
        whyMatch.push('Your lighter build is ideal for endurance');
      }
    }

    // Tennis scoring
    if (sport.name === 'Tennis') {
      if (user.height > 175 && user.height < 195) {
        score += 25;
        whyMatch.push('Your height is perfect for tennis');
      }
      if (user.wingspan > user.height + 3) {
        score += 20;
        whyMatch.push('Your reach gives you an advantage');
      }
      if (user.weight > 65 && user.weight < 90) {
        score += 25;
        whyMatch.push('Your weight is ideal for power and agility');
      }
      if (user.height > 180) {
        score += 15;
        whyMatch.push('Your height helps with serve power');
      }
      if (user.weight > 70 && user.weight < 85) {
        score += 15;
        whyMatch.push('Your build is perfect for court coverage');
      }
    }

    // Track & Field scoring
    if (sport.name === 'Track & Field') {
      if (user.height > 180) {
        score += 20;
        whyMatch.push('Your height is great for sprinting');
      }
      if (user.weight > 70 && user.weight < 95) {
        score += 25;
        whyMatch.push('Your weight is ideal for explosive power');
      }
      if (user.height > 190) {
        score += 15;
        whyMatch.push('Your height is advantageous for many events');
      }
      if (user.weight < 80) {
        score += 20;
        whyMatch.push('Your lighter build is great for speed');
      }
      if (user.height > 185) {
        score += 20;
        whyMatch.push('Your height is perfect for jumping events');
      }
    }

    // Swimming scoring
    if (sport.name === 'Swimming') {
      if (user.height > 180) {
        score += 25;
        whyMatch.push('Your height is ideal for swimming');
      }
      if (user.wingspan > user.height + 5) {
        score += 20;
        whyMatch.push('Your long arms are perfect for swimming');
      }
      if (user.weight > 70 && user.weight < 95) {
        score += 20;
        whyMatch.push('Your weight is ideal for buoyancy and power');
      }
      if (user.height > 185) {
        score += 15;
        whyMatch.push('Your height gives you a natural advantage');
      }
      if (user.wingspan > user.height + 3) {
        score += 20;
        whyMatch.push('Your reach is perfect for stroke efficiency');
      }
    }

    // MMA scoring
    if (sport.name === 'MMA') {
      if (user.weight > 65 && user.weight < 100) {
        score += 25;
        whyMatch.push('Your weight is suitable for multiple weight classes');
      }
      if (user.height > 170 && user.height < 190) {
        score += 20;
        whyMatch.push('Your height is ideal for MMA');
      }
      if (user.wingspan > user.height) {
        score += 15;
        whyMatch.push('Your reach is advantageous for striking');
      }
      if (user.weight > 70 && user.weight < 90) {
        score += 20;
        whyMatch.push('Your build is perfect for most weight classes');
      }
      if (user.height > 175) {
        score += 20;
        whyMatch.push('Your height gives you reach advantages');
      }
    }

    // Gymnastics scoring
    if (sport.name === 'Gymnastics') {
      if (user.height < 170) {
        score += 30;
        whyMatch.push('Your height is perfect for gymnastics');
      }
      if (user.weight < 70) {
        score += 25;
        whyMatch.push('Your lighter build is ideal for acrobatics');
      }
      if (user.height < 160) {
        score += 20;
        whyMatch.push('Your compact build is great for tumbling');
      }
      if (user.weight < 60) {
        score += 20;
        whyMatch.push('Your weight is perfect for aerial maneuvers');
      }
      if (user.height < 175) {
        score += 15;
        whyMatch.push('Your height is suitable for most events');
      }
    }

    if (score > 0) {
      recommendations.push({
        sport: sport.name,
        score: Math.min(score, 100),
        description: sport.description,
        icon: sport.icon,
        whyMatch
      });
    }
  });

  return recommendations.sort((a, b) => b.score - a.score).slice(0, 3);
}

export function generateAnalysis(user: BodyMeasurements, topAthletes: AthleteMatch[], topSports: SportRecommendation[]): string {
  const avgHeight = topAthletes.reduce((sum, match) => sum + match.athlete.height, 0) / topAthletes.length;
  const avgWeight = topAthletes.reduce((sum, match) => sum + match.athlete.weight, 0) / topAthletes.length;
  
  let analysis = `Based on your measurements (${user.height}cm, ${user.weight}kg, ${user.wingspan}cm wingspan), `;
  
  if (user.height > avgHeight + 5) {
    analysis += "your height gives you a significant advantage in sports requiring reach and verticality. ";
  } else if (user.height < avgHeight - 5) {
    analysis += "your compact build is perfect for sports requiring agility and quick movements. ";
  } else {
    analysis += "your well-proportioned build makes you versatile across multiple sports. ";
  }

  if (user.wingspan > user.height + 10) {
    analysis += "Your exceptional wingspan is a major asset for sports requiring reach and leverage. ";
  }

  if (topSports.length > 0) {
    analysis += `Your body type is most similar to ${topSports[0].sport} athletes, with a ${topSports[0].score}% compatibility score. `;
  }

  if (topAthletes.length > 0) {
    analysis += `You share the most physical similarities with ${topAthletes[0].athlete.name}, a ${topAthletes[0].athlete.sport} athlete.`;
  }

  return analysis;
}

export const bodyParts: BodyPart[] = [
  {
    id: 'head',
    name: 'Head',
    measurement: 'height',
    position: { x: 50, y: 5 },
    size: { width: 20, height: 15 },
    description: 'Overall height measurement'
  },
  {
    id: 'shoulders',
    name: 'Shoulders',
    measurement: 'shoulderWidth',
    position: { x: 40, y: 20 },
    size: { width: 40, height: 15 },
    description: 'Shoulder width measurement'
  },
  {
    id: 'torso',
    name: 'Torso',
    measurement: 'weight',
    position: { x: 45, y: 35 },
    size: { width: 30, height: 25 },
    description: 'Weight and core measurements'
  },
  {
    id: 'arms',
    name: 'Arms',
    measurement: 'wingspan',
    position: { x: 20, y: 25 },
    size: { width: 15, height: 30 },
    description: 'Wingspan and arm length'
  },
  {
    id: 'waist',
    name: 'Waist',
    measurement: 'waist',
    position: { x: 47, y: 50 },
    size: { width: 26, height: 10 },
    description: 'Waist measurement'
  },
  {
    id: 'hips',
    name: 'Hips',
    measurement: 'hip',
    position: { x: 45, y: 60 },
    size: { width: 30, height: 15 },
    description: 'Hip width measurement'
  },
  {
    id: 'legs',
    name: 'Legs',
    measurement: 'height',
    position: { x: 48, y: 75 },
    size: { width: 24, height: 20 },
    description: 'Leg length and strength'
  }
];
