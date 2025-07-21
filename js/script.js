// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize scroll animations
    initScrollAnimations();

    // Initialize button effects
    initButtonEffects();

    // Initialize media controls
    initMediaControls();
});

// Scroll animations
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);

    // Observe all section elements
    document.querySelectorAll('.section > *').forEach(el => {
        observer.observe(el);
    });
}

// Button effects
function initButtonEffects() {
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            // Create ripple effect
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple 0.6s ease-out;
                pointer-events: none;
            `;

            this.appendChild(ripple);

            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
}

// Media controls
function initMediaControls() {
    const videos = document.querySelectorAll('video');
    const gifs = document.querySelectorAll('img[src$=".gif"]');

    // Video controls
    videos.forEach(video => {
        const container = video.closest('.gallery-item');

        // Add play/pause on click
        video.addEventListener('click', function() {
            if (this.paused) {
                this.play();
            } else {
                this.pause();
            }
        });

        // Add hover controls
        container.addEventListener('mouseenter', function() {
            video.play();
        });

        container.addEventListener('mouseleave', function() {
            video.pause();
        });
    });

    // GIF lazy loading
    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                }
                imageObserver.unobserve(img);
            }
        });
    });

    gifs.forEach(gif => {
        imageObserver.observe(gif);
    });
}

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Handle window resize
window.addEventListener('resize', debounce(() => {
    // Re-initialize any responsive elements if needed
    console.log('Window resized');
}, 250));

// Handle page visibility change
document.addEventListener('visibilitychange', function() {
    const videos = document.querySelectorAll('video');

    if (document.hidden) {
        // Pause all videos when page is hidden
        videos.forEach(video => {
            if (!video.paused) {
                video.pause();
                video.dataset.wasPlaying = 'true';
            }
        });
    } else {
        // Resume videos that were playing
        videos.forEach(video => {
            if (video.dataset.wasPlaying === 'true') {
                video.play();
                video.removeAttribute('data-was-playing');
            }
        });
    }
});

// Performance optimization: Intersection Observer for media loading
function initLazyLoading() {
    const lazyElements = document.querySelectorAll('[data-src]');

    const lazyObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;

                if (element.tagName === 'IMG') {
                    element.src = element.dataset.src;
                } else if (element.tagName === 'VIDEO') {
                    element.src = element.dataset.src;
                    element.load();
                }

                element.removeAttribute('data-src');
                lazyObserver.unobserve(element);
            }
        });
    }, {
        rootMargin: '50px'
    });

    lazyElements.forEach(element => {
        lazyObserver.observe(element);
    });
}