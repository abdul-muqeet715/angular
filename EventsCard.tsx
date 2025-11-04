import {Card} from './ui/card';
import {Badge} from './ui/badge';
import {Calendar, MapPin, Users} from 'lucide-react';

interface EventCardProps {
    className ?: string;
    onEventClick ?: (type: 'event' | 'weather', id: string) => void;
}

export function EventCard({ className, onEventClick}: EventCardProps) {
    const currentEvents = [
        {
            id : '1',
            type: 'Holiday',
            name: 'Martin Luther King Day',
            impact: 'High',
            date: '2024-01-15',
            affectedCashpoints: 23,
            amountChange: 150000,
            time: 'All Day',
            location: 'Nationwide'
        },
        {
             id : '2',
            type: 'Local Event',
            name: 'Downtown festival',
            impact: 'medium',
            date: '2024-01-16',
            affectedCashpoints: 8,
            amountChange: 500,
            time: 'All Day',
            location: 'Downtown area'
        }
    ];

    const getImpactColor = (impact: string) => {
        switch(impact) {
            case 'high' : return 'bg-red-100 text-red-800 border-red-200';
            case 'medium': return 'bg-yello-100 text-yello-800 border-yellow-200';
            case 'low': return 'bg-green-100 text-green-800 border-green-200';
            default: return 'bg-gray-100 text-gray-800 border-gray-200';
        }

    };

    return {
        <Card className={`p-4 ${className}`}>
         <div className = "flex items-center gap-2 mb-4">
            <Calendar className="w5 h-5 text-primary" />
            <h3 className = "font-medium">Current Events</h3>
         </div>

         
    }
}