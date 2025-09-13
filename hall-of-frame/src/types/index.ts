export interface BodyMeasurements {
  height: number; // in cm
  weight: number; // in kg
  wingspan: number; // in cm
  shoulderWidth?: number; // in cm
  waist?: number; // in cm
  hip?: number; // in cm
}

export interface Athlete {
  id: string;
  name: string;
  sport: string;
  position?: string;
  height: number; // in cm
  weight: number; // in kg
  wingspan: number; // in cm
  shoulderWidth?: number; // in cm
  waist?: number; // in cm
  hip?: number; // in cm
  imageUrl?: string;
  description?: string;
  achievements?: string[];
}

export interface SportRecommendation {
  sport: string;
  score: number; // 0-100
  description: string;
  icon: string;
  whyMatch: string[];
}

export interface AthleteMatch {
  athlete: Athlete;
  similarityScore: number; // 0-100
  matchingTraits: string[];
}

export interface UserResults {
  topSports: SportRecommendation[];
  topAthletes: AthleteMatch[];
  userMeasurements: BodyMeasurements;
  analysis: string;
}

export interface BodyPart {
  id: string;
  name: string;
  measurement: keyof BodyMeasurements;
  position: { x: number; y: number };
  size: { width: number; height: number };
  description: string;
}
