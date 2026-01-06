import { useState } from 'react';
import './SourceCard.css';

/**
 * Displays a legal source citation with expandable content
 */
export default function SourceCard({ source, index }) {
    const [isExpanded, setIsExpanded] = useState(false);

    const {
        law_title = 'Unknown Law',
        article_title = '',
        text = '',
        score = 0,
        highlight_path = '',
    } = source;

    const relevancePercent = Math.round(score * 100);

    return (
        <div className="source-card">
            <button
                className="source-card__header"
                onClick={() => setIsExpanded(!isExpanded)}
                aria-expanded={isExpanded}
            >
                <span className="source-card__index">{index + 1}</span>
                <div className="source-card__info">
                    <h4 className="source-card__title">{law_title}</h4>
                    {article_title && (
                        <span className="source-card__article">{article_title}</span>
                    )}
                </div>
                <div className="source-card__score">
                    <div
                        className="source-card__score-bar"
                        style={{ width: `${relevancePercent}%` }}
                    />
                    <span className="source-card__score-text">{relevancePercent}%</span>
                </div>
                <svg
                    className={`source-card__chevron ${isExpanded ? 'source-card__chevron--open' : ''}`}
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                >
                    <path d="M6 9l6 6 6-6" />
                </svg>
            </button>

            {isExpanded && (
                <div className="source-card__content">
                    {highlight_path && (
                        <p className="source-card__path">{highlight_path}</p>
                    )}
                    <p className="source-card__text">{text}</p>
                </div>
            )}
        </div>
    );
}
